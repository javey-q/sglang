## Reviewer summary

This PR is complementary to FlashInfer's native [Block Extend Attention](https://github.com/flashinfer-ai/flashinfer/pull/2722) work, rather than a duplicate of it. That upstream work provides the operator-level primitive; this PR provides the SGLang scheduler/runtime changes required to turn larger dLLM prefill chunks into a correct execution mode:

- decouple pure prefill from fixed-size decode, so a long prompt can advance by multiple dLLM blocks per scheduling round while decode remains one fixed block;
- fix two phase-selection ordering bugs in `determine_dllm_phase()` / request admission, where prefix-cache matching or late incoming-request preparation could classify a decode block as prefill for the current round;
- validate the scheduling and attention behavior with unit/kernel tests and H200 end-to-end measurements (up to `+491%` input throughput at 16k context in the included matrix).

The explicit FlashInfer custom attention mask is deliberately an interim compatibility path for block-diffusion attention. It makes the scheduling improvement usable before native Block Extend Attention is released in the FlashInfer version supported by SGLang. Once that operator is available, this mask path can be replaced with the native operator without changing the scheduler-facing prefill/decode separation introduced here.

Reviewers may therefore focus on the scheduler phase boundaries and request lifecycle independently of the temporary attention implementation.

## Motivation

This PR addresses the long-prefill optimization proposed in #24644 and #24645.

Previously, dLLM prefill and decode shared the same `block_size` (typically 32). As a result, a long prompt could advance by only one decode block per scheduler round, causing excessive scheduling overhead and underutilizing the GPU.

This PR decouples the maximum prefill chunk size from the fixed decode block size:

- decode still processes exactly one `block_size` block per request;
- pure prefill can process multiple aligned blocks in one scheduler round;
- the non-aligned prompt tail is handled together with masks by the following decode block.



### Execution-mode separation

Pure-prefill requests intentionally use the regular `ForwardMode.EXTEND`, while decode requests continue to use `ForwardMode.DLLM_EXTEND`. Pure prefill has a dynamic extend length and commits only real prompt KV; decode must preserve the fixed `block_size` dLLM semantics and its dedicated CUDA Graph path.

**This separation is a key correctness point for review.** The scheduler must determine the final request phase before it constructs the phase-aware `PrefillAdder` and selects the forward mode. A stale or prematurely selected prefill phase could incorrectly send a decode request through regular `EXTEND` instead of `DLLM_EXTEND`.

### Phase-selection correctness problems addressed

This PR also fixes two ordering issues exposed by multi-block prefill:

1. `determine_dllm_phase()` previously ran with stale `prefix_indices`. Prefix-cache matching could update the matched prefix without reclassifying the request, so a prompt-tail-plus-mask block could remain incorrectly classified as pure prefill. Phase determination now runs after cache matching has finalized `prefix_indices`.
2. `get_new_batch_dllm()` previously selected `is_prefill`, created the adder, and selected the forward mode before incoming requests were fully prepared. Even if `process_dllm_incoming_reqs()` later changed a request to decode, the current round had already been configured as prefill. All managed dLLM requests are now collected and prepared before phase selection, so the adder and forward mode are constructed from the final cache-matched phases.

Related to #24644, #24645.

## Modifications



### Configurable multi-block prefill

- Add an optional `prefill_block_size` field to `--dllm-algorithm-config`.
- Add `--dllm-prefill-block-size` as a command-line override for the YAML `prefill_block_size` value.
- Default `prefill_block_size` to `block_size` to preserve the existing behavior.
- Require `prefill_block_size >= block_size` and divisibility by `block_size`.
- Require users to explicitly pass `--attention-backend flashinfer` when `prefill_block_size > block_size`, because multi-block dLLM prefill currently depends on FlashInfer's custom attention-mask support.
- Reject unsupported attention backend combinations at startup instead of silently using incorrect attention semantics. The existing single-block behavior (`prefill_block_size == block_size`) is unaffected by this requirement.

Example:

```yaml
block_size: 32
prefill_block_size: 1024
threshold: 0.95
```

The YAML value can be overridden without editing the configuration file:

```bash
python -m sglang.launch_server ... \
  --dllm-prefill-block-size 1024 \
  --attention-backend flashinfer
```

### Phase-aware dLLM scheduling

- Separate pure-prefill scheduling from fixed-size decode scheduling.
- Fetch incoming requests and prepare all managed dLLM requests before selecting the round phase.
- Re-run phase determination after prefix-cache matching updates `prefix_indices`, preventing stale-prefix phase classification.
- Use phase-aware scheduler budgets:
  - prefill: `max_running_requests * prefill_block_size`;
  - decode: `max_running_requests * block_size`.
- Perform prefix-cache matching before determining the dLLM phase.
- Construct the phase-aware adder and select `EXTEND` versus `DLLM_EXTEND` only after the final request phases are known.
- Preserve the incoming request lifecycle until admission by `PrefillAdder`.
- Prefer staging requests over incoming requests within the same phase.
- Prevent unadmitted requests from advancing `dllm_block_offset`.
- Align pure-prefill frontiers to `block_size`.

For example, with `block_size=32` and a 300-token prompt, pure prefill advances to token 288. The remaining 12 prompt tokens and 20 masks are processed by the next 32-token decode block.

### Block-wise prefill attention

Multi-block dLLM prefill uses the following visibility rule, where `B` is the dLLM block size:

```text
visible(q, k) = floor(k / B) <= floor(q / B)
```

The custom-mask implementation in this PR is an interim solution until FlashInfer's native [Block Extend Attention support](https://github.com/flashinfer-ai/flashinfer/pull/2722) is merged, released, and available in SGLang's supported FlashInfer version. Once that native operator is available, this path can use its block-expanding mask semantics directly and avoid constructing and passing an additional attention mask. Migrating to the native operator is therefore the intended follow-up rather than keeping the explicit custom-mask path permanently.

This provides:

- bidirectional attention within the same dLLM block;
- visibility into all preceding blocks;
- isolation from future blocks.

The FlashInfer implementation includes:

- a request-wise flattened custom block mask;
- a fast path without a custom mask when every extend fits in one block;
- ragged prefill with prefix cascade:
  - masked ragged attention over the current extend;
  - unmasked paged attention over the cached prefix;
  - output/LSE combination through `merge_state`;
- a paged `E x (P + E)` correctness fallback when ragged prefill is disabled.

Masked and unmasked ragged prefill use separate persistent FlashInfer wrappers. The custom-mask wrapper is pinned to FA2, preventing an unmasked FA3 plan from being incorrectly reused for a later masked request on SM90.

Unsupported combinations such as multi-block custom masks with sliding-window dispatch or multi-item scoring fail explicitly.

### Breakable prefill CUDA Graph

- Run pure dLLM prefill through `ForwardMode.EXTEND`.
- Allow reuse of Breakable prefill CUDA Graphs when:
  - CUDA and FlashInfer are used;
  - the prefill graph backend is Breakable;
  - the token count exactly matches a captured bucket;
  - no unsupported input mode is active.
- Fall back to eager execution when these conditions are not satisfied.
- Propagate dLLM prefill metadata through graph replay so the same block-wise mask is reconstructed.

Exact bucket matching is required because upward padding would change dLLM bidirectional attention and KV-write semantics.

### Tests and documentation

- Add scheduler/configuration tests for:
  - phase-aware budgets;
  - aligned and non-aligned prompt boundaries;
  - incoming request lifecycle and prefix-cache phase detection;
  - fixed decode blocks;
  - request admission and offset updates;
  - FlashInfer backend validation;
  - Breakable CUDA Graph capability checks.
- Add FlashInfer attention tests for:
  - block-mask construction;
  - heterogeneous batches;
  - single-block fast path;
  - non-aligned prefixes;
  - masked/unmasked wrapper isolation;
  - ragged output against a dense reference;
  - prefix cascade output and LSE parity;
  - future-block isolation;
  - paged fallback parity.
- Document the new server argument and scheduling/attention design.



## Accuracy Tests

Validation environment:


| Item                     | Value                       |
| ------------------------ | --------------------------- |
| GPU                      | 2x NVIDIA H200 143 GB       |
| Model                    | `inclusionAI/LLaDA2.0-mini` |
| PyTorch                  | `2.11.0+cu128`              |
| FlashInfer               | `0.6.12`                    |
| dLLM block size          | 32                          |
| Multi-block prefill size | 1024                        |




### Unit and kernel tests

```bash
CUDA_VISIBLE_DEVICES=1 FLASHINFER_WORKSPACE_BASE=/tmp \
  python test/registered/unit/layers/test_flashinfer_dllm_prefill_mask.py -v

CUDA_VISIBLE_DEVICES=1 \
  python test/registered/unit/managers/test_prefill_adder.py -v
```

Validated attention-level results:


| Test                                                            | Result                                     |
| --------------------------------------------------------------- | ------------------------------------------ |
| Production-shaped bf16 GQA ragged attention vs. dense reference | PASS                                       |
| Prefix cascade output and LSE vs. dense reference               | PASS                                       |
| Future-block K/V isolation                                      | PASS                                       |
| Paged fallback vs. dense reference                              | PASS                                       |
| Multi-block eager long-prompt regression                        | PASS; no `operation not supported` failure |
| Multi-block eager vs. Breakable, prompt lengths 31-2060         | Identical output IDs                       |


For the prefix-cascade validation with `P=64`, `E=128`, and `B=32`:

- maximum output error: `5.3e-4`;
- maximum LSE error: `7e-5`;
- future-block early-output delta: `0`;
- paged fallback maximum error: `2.7e-4`.



### GSM8K quality

GSM8K was evaluated with 200 examples, `temperature=0`, TP=1, and radix cache disabled.


| Configuration              | Score |
| -------------------------- | ----- |
| Main baseline              | 0.895 |
| Current single-block eager | 0.895 |
| Multi-block eager          | 0.915 |
| Multi-block Breakable      | 0.910 |


Acceptance results:

```text
multi_eager - main            = +0.020
multi_eager - single_block    = +0.020
multi_breakable - multi_eager = -0.005
```

Both the multi-block eager and Breakable configurations pass the configured absolute and relative quality thresholds.

Single-block and multi-block execution are not expected to be token-identical for longer prompts because changing the prefill chunk boundaries also changes kernel shapes, KV submission boundaries, and floating-point accumulation. This difference was also observed after forcing both paths to FA2. The end-to-end GSM8K quality evaluation shows no accuracy regression.

## Speed Tests and Profiling

### GSM8K latency (H200, TP=1, 200 examples)

| Configuration | Latency (s) | Output throughput (tok/s) |
| --- | ---: | ---: |
| Main baseline | 65.8 | 381.0 |
| Current single-block eager | 121.6 | 205.2 |
| Multi-block eager | 58.6 | 435.0 |
| Multi-block Breakable | 47.1 | 540.7 |

Compared with the existing single-block eager configuration:

- multi-block eager improves output throughput from `205.2` to `435.0 tok/s` (`2.12x`);
- multi-block Breakable improves output throughput to `540.7 tok/s` (`2.63x`);
- multi-block Breakable reduces evaluation latency from `121.6s` to `47.1s`.

Compared with multi-block eager, Breakable prefill graph replay improves output throughput by approximately `1.24x`.

### Prefill serving matrix (H200, TP=1, exact Breakable buckets)

Full matrix under `benchmark_results/pr_prefill_perf_matrix_20260720/`:

- configs: `main` (`block_size=32`) + current-branch `prefill_block_size ∈ {32,128,512,1024,2048,4096}`
- input lengths: `128, 256, 512, 1024, 4096, 8192, 16384`
- `num_prompts=16`, `output_len=64`, `max_concurrency=1`, fixed seeds (`seed = 2000 + input_len`)

Input token throughput (tok/s):

| input | main | pbs32 | pbs128 | pbs512 | pbs1024 | pbs2048 | pbs4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 528 | 416 | 451 | 438 | 441 | 441 | 442 |
| 256 | 917 | 774 | 897 | 846 | 850 | 847 | 849 |
| 512 | 1832 | 1545 | 1967 | 1968 | 1982 | 1989 | 1977 |
| 1024 | 2563 | 2251 | 2975 | 3245 | 3250 | 3268 | 3299 |
| 4096 | 4784 | 4251 | 8561 | 10759 | 12420 | 12810 | 12320 |
| 8192 | 5517 | 4543 | 11845 | 18841 | 23248 | 25174 | 25671 |
| 16384 | 5193 | 3757 | 13103 | 22751 | 27713 | 30709 | 29577 |

vs `main` input-throughput delta at long context:

| input | pbs1024 | pbs2048 | pbs4096 |
| ---: | ---: | ---: | ---: |
| 4k | +160% | +168% | +158% |
| 8k | +321% | +356% | +365% |
| 16k | +434% | +491% | +470% |

Mean TTFT (ms) at 16k: `main 2996` → `pbs1024 410` / `pbs2048 351` / `pbs4096 365`.

Notes:

- Current-branch `pbs32` is slightly slower than `main` (expected overhead of the new phase-aware path with the same chunk size).
- Gains grow with prompt length; sweet spot on this setup is about `prefill_block_size=2048` for 8k–16k.
- Prefill CUDA-graph miss count was `0` across completed sweeps; no OOM on H200 with `mem-fraction-static` 0.75–0.85.

## Checklist

- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [ ] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).



## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
  - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`.
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
