import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


class _LmHeadWasCalled(Exception):
    pass


class _ExplodingLmHead:
    """Any use of the LM head fails loudly instead of silently costing time."""

    def __call__(self, *args, **kwargs):
        raise _LmHeadWasCalled()

    def __getattr__(self, name):
        raise _LmHeadWasCalled()


def _dllm_prefill_forward_batch(**overrides):
    """Minimal stand-in for a dLLM pure-prefill ForwardBatch.

    Only the fields LogitsMetadata.from_forward_batch reads are provided, so
    the test also pins that the is_dllm_prefill flag is actually plumbed
    through -- if that wiring is dropped the skip silently stops happening and
    nothing else in the suite notices.
    """
    fields = dict(
        forward_mode=ForwardMode.EXTEND,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        next_token_logits_buffer=None,
        return_logprob=False,
        extend_seq_lens=torch.tensor([8]),
        extend_seq_lens_cpu=[8],
        extend_logprob_start_lens_cpu=[0],
        top_logprobs_nums=[0],
        token_ids_logprobs=[None],
        extend_input_logprob_token_ids_gpu=None,
        is_prefill_only=False,
        is_dllm_prefill=True,
        global_num_tokens_gpu=None,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_dp_buffer_len=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_tokens_for_logprob_gpu=None,
        mm_input_embeds=None,
        multi_item_delimiter_indices=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestDllmPrefillSkipsLmHead(CustomTestCase):
    """dLLM pure prefill only commits prompt KV.

    SchedulerDllmMixin.process_batch_result_dllm returns before reading
    logits_output, so every LM-head / logits-processor op that forward() runs
    for such a batch is discarded work. The skip has no output-visible effect
    -- which is exactly why it needs a test: a regression here costs latency
    silently, and no accuracy or parity check would go red.
    """

    def _took_skip_path(self, forward_batch) -> bool:
        """Did forward() return the no-logits shortcut?

        The shortcut returns before any parameterized submodule is touched, so
        a bare LogitsProcessor is enough to exercise it. Anything that falls
        through to the real LM-head path needs the fully-constructed module and
        raises instead -- which is the observation we want, and the exploding
        lm_head keeps a completed real path from being mistaken for the
        shortcut.
        """
        processor = object.__new__(LogitsProcessor)
        try:
            output = LogitsProcessor.forward(
                processor,
                input_ids=None,
                hidden_states=torch.zeros(8, 4),
                lm_head=_ExplodingLmHead(),
                logits_metadata=LogitsMetadata.from_forward_batch(forward_batch),
            )
        except Exception:
            return False
        return output.next_token_logits is None

    def test_pure_prefill_skips_the_lm_head(self):
        self.assertTrue(self._took_skip_path(_dllm_prefill_forward_batch()))

    def test_decode_batch_keeps_the_normal_path(self):
        """Decode carries is_dllm_prefill=False. Skipping there would drop the
        logits generation actually depends on."""
        self.assertFalse(
            self._took_skip_path(_dllm_prefill_forward_batch(is_dllm_prefill=False))
        )

    def test_hidden_state_capture_keeps_the_normal_path(self):
        """enable_return_hidden_states / speculative decoding need real hidden
        states, so the shortcut must not swallow those batches."""
        self.assertFalse(
            self._took_skip_path(
                _dllm_prefill_forward_batch(capture_hidden_mode=CaptureHiddenMode.FULL)
            )
        )

    def test_requested_logprobs_keep_the_normal_path(self):
        self.assertFalse(
            self._took_skip_path(_dllm_prefill_forward_batch(return_logprob=True))
        )


if __name__ == "__main__":
    unittest.main()
