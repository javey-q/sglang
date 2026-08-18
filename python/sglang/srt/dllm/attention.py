from typing import Optional, Sequence

import torch


def build_dllm_prefill_blockwise_mask(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    block_size: int,
    device: torch.device,
    *,
    include_prefix: bool,
) -> Optional[torch.Tensor]:
    """Build FlashInfer's flattened per-request dLLM prefill mask.

    Tokens in the same dLLM block are mutually visible, while a query may only
    attend to its own block and earlier blocks. With ``include_prefix=False``
    the key dimension covers only the current ragged extend chunk. With
    ``include_prefix=True`` it covers the paged ``prefix + extend`` sequence.

    Returns ``None`` when no request's extend range crosses a block boundary,
    for which non-causal attention already has the desired visibility.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if len(prefix_lens) != len(extend_lens):
        raise ValueError(
            "prefix_lens and extend_lens must have the same length: "
            f"{len(prefix_lens)} != {len(extend_lens)}"
        )
    if any(
        prefix_len < 0 or extend_len < 0
        for prefix_len, extend_len in zip(prefix_lens, extend_lens)
    ):
        raise ValueError("prefix and extend lengths must be non-negative")
    needs_mask = any(
        extend_len > 0
        and prefix_len // block_size != (prefix_len + extend_len - 1) // block_size
        for prefix_len, extend_len in zip(prefix_lens, extend_lens)
    )
    if not needs_mask:
        return None

    mask_parts = []
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        query_positions = prefix_len + torch.arange(
            extend_len, device=device, dtype=torch.int64
        )
        if include_prefix:
            key_positions = torch.arange(
                prefix_len + extend_len, device=device, dtype=torch.int64
            )
        else:
            key_positions = prefix_len + torch.arange(
                extend_len, device=device, dtype=torch.int64
            )

        query_blocks = torch.div(query_positions, block_size, rounding_mode="floor")
        key_blocks = torch.div(key_positions, block_size, rounding_mode="floor")
        mask_parts.append((key_blocks[None, :] <= query_blocks[:, None]).flatten())

    if not mask_parts:
        return None
    return torch.cat(mask_parts)


def build_dllm_prefill_packed_mask(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    block_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Emit the ragged blockwise mask already bit-packed, skipping the bool form.

    Every row of the ragged mask is a *prefix* of ones: query ``i`` sees every
    key up to the end of its own dLLM block and nothing after. So the packed
    bytes follow directly from that prefix length, and neither the O(q x k)
    bool tensor nor FlashInfer's ``segment_packbits`` has to run --
    ``segment_packbits`` is ~94% of ``plan()`` at a 4096-token chunk.

    Returns ``None`` when a chunk length is not a multiple of 8, because then
    rows (and segment offsets) are not byte-aligned and the packed layout the
    kernel expects cannot be built this way; callers fall back to the bool mask.

    Layout matches ``segment_packbits(..., bitorder="little")``: byte ``b`` of
    row ``i`` holds keys ``8b .. 8b+7`` in its low-to-high bits, and segments
    are concatenated in request order.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if len(prefix_lens) != len(extend_lens):
        raise ValueError(
            "prefix_lens and extend_lens must have the same length: "
            f"{len(prefix_lens)} != {len(extend_lens)}"
        )
    if any(
        prefix_len < 0 or extend_len < 0
        for prefix_len, extend_len in zip(prefix_lens, extend_lens)
    ):
        raise ValueError("prefix and extend lengths must be non-negative")
    if any(extend_len % 8 for extend_len in extend_lens):
        return None

    parts = []
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        if extend_len == 0:
            continue
        block_offset = prefix_len % block_size
        query_index = torch.arange(extend_len, device=device, dtype=torch.int64)
        # Number of leading keys visible to each query: everything up to the
        # end of the block that query sits in, clamped to the chunk.
        visible = torch.clamp(
            ((query_index + block_offset) // block_size + 1) * block_size
            - block_offset,
            max=extend_len,
        )
        byte_index = torch.arange(extend_len // 8, device=device, dtype=torch.int64)
        bits_in_byte = torch.clamp(
            visible[:, None] - byte_index[None, :] * 8, min=0, max=8
        )
        parts.append(((1 << bits_in_byte) - 1).to(torch.uint8).flatten())

    if not parts:
        return torch.zeros(0, dtype=torch.uint8, device=device)
    return torch.cat(parts)


def dllm_prefill_packed_mask_indptr(
    extend_lens: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Byte offsets of each request's packed mask segment.

    FlashInfer's kernel indexes ``custom_mask`` with byte offsets -- what
    ``segment_packbits`` returns. ``plan(packed_custom_mask=...)`` instead
    stores ``_compute_mask_indptr``, which is in *bits*, so every request after
    the first reads 8x past its segment. Callers that own ``mask_indptr_buf``
    overwrite it with this after planning.
    """
    sizes = torch.tensor(
        [extend_len * extend_len // 8 for extend_len in extend_lens],
        dtype=torch.int32,
        device=device,
    )
    indptr = torch.zeros(len(extend_lens) + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(sizes, 0)
    return indptr


def build_dllm_prefill_cuda_graph_mask(
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Ragged blockwise prefill mask that is never ``None``.

    Under a full CUDA graph the ragged wrapper owns a static ``custom_mask_buf``,
    and FlashInfer selects its mask mode from whether that buffer was
    initialized -- not from whether a mask was passed to ``plan()``. Skipping
    the mask on a replay therefore leaves the kernel reading whatever the buffer
    last held instead of falling back to non-causal attention. When no chunk
    crosses a dLLM block boundary, full visibility over the ragged chunk is
    exactly the mask that plain non-causal attention would have applied.
    """
    mask = build_dllm_prefill_blockwise_mask(
        prefix_lens,
        extend_lens,
        block_size,
        device,
        include_prefix=False,
    )
    if mask is not None:
        return mask
    numel = sum(extend_len * extend_len for extend_len in extend_lens)
    return torch.ones(numel, dtype=torch.bool, device=device)
