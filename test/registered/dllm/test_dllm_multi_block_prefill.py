from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

NEEDLE = "4271"
# Long enough to span several 256-token prefill chunks, with the needle sitting
# in an early chunk so retrieval depends on KV committed by an earlier round.
LONG_PROMPT = (
    "Read the notes and answer the question.\n"
    f"Note 1: The access code for the east gate is {NEEDLE}.\n"
    + "Note: The quick brown fox jumps over the lazy dog. " * 120
    + "\nQuestion: What is the access code for the east gate? Answer:"
)
# Comfortably inside one dLLM block, so every chunk size prefills it identically.
SHORT_PROMPTS = [
    "The capital of France is",
    "Q: What is 12 times 13? A:",
]


class TestMultiBlockPrefill(CustomTestCase):
    """End-to-end guard for ``--dllm-prefill-block-size`` > block_size.

    Multi-block prefill is the only path that runs pure prefill as a normal
    EXTEND batch: FlashInfer's blockwise custom mask replaces the fixed-block
    attention and exact-token prefill CUDA graph buckets replace the decode
    graph. A wrong mask, block offset or bucket corrupts the prompt KV, which
    shows up as failed long-context retrieval.

    Bit-identical output across chunk sizes is deliberately *not* asserted:
    chunking changes the attention reduction order, and measurements on H200 /
    LLaDA2.0-mini show prompts that span several rounds reword their answer
    between every chunk size (32/64/128/256 all differ pairwise) while GSM8K
    accuracy stays flat (0.925 vs 0.920 at 200 examples). Prompts that fit in
    one chunk do stay identical, which is what the short-prompt check pins.
    """

    model = "inclusionAI/LLaDA2.0-mini"
    base_url = DEFAULT_URL_FOR_TEST

    def _collect_outputs(self, prefill_block_size: int):
        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "1",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "LowConfidence",
            "--no-dllm-fdfo",
            "--cuda-graph-bs",
            "1",
            "--dllm-prefill-block-size",
            str(prefill_block_size),
        ]

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            outputs = []
            for prompt in [LONG_PROMPT, *SHORT_PROMPTS]:
                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": 64,
                        "temperature": 0,
                    },
                    timeout=120,
                )
                outputs.append(response.json()["choices"][0]["text"])
            return outputs
        finally:
            kill_process_tree(process.pid)

    def test_multi_block_prefill_matches_single_block(self):
        # block_size is 32 for LLaDA2.0-mini, so 32 keeps the single-block
        # DLLM_EXTEND path and 256 opts into multi-block EXTEND prefill.
        single_block = self._collect_outputs(prefill_block_size=32)
        multi_block = self._collect_outputs(prefill_block_size=256)

        for label, outputs in (
            ("single-block", single_block),
            ("multi-block", multi_block),
        ):
            self.assertIn(
                NEEDLE,
                outputs[0],
                f"{label} prefill lost the long-context needle; the prompt KV "
                "is wrong (blockwise custom mask, dLLM block offsets, or the "
                "exact-token graph buckets).",
            )

        self.assertEqual(
            multi_block[1:],
            single_block[1:],
            "Prompts that fit in a single chunk are prefilled by one round in "
            "both configurations, so their output must be identical.",
        )


if __name__ == "__main__":
    unittest.main()
