from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from PIL import Image

from layers.qwen_bedrock_provider import QwenBedrockMantleHTRProvider


def _response(content: str, finish_reason: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _payload(text: str = "अग्निम्") -> str:
    return json.dumps(
        {
            "script_guess": "Devanagari",
            "candidates": [
                {"text": text, "confidence": 0.9},
                {"text": text + "x", "confidence": 0.7},
            ],
            "uncertainties": [],
            "visual_observations": "test",
        },
        ensure_ascii=False,
    )


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected extra completion call.")
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class ProviderCTruncationRetryTest(unittest.TestCase):
    def test_length_retries_once_at_1600(self):
        truncated = (
            '{"script_guess":"Devanagari","candidates":'
            '[{"text":"अ","confidence":0.9}],'
            '"uncertainties":[{"span":"अ","alternatives":["अ"'
        )
        client = _FakeClient(
            [
                _response(truncated, "length"),
                _response(_payload(), "stop"),
            ]
        )
        provider = QwenBedrockMantleHTRProvider()
        provider._client = lambda: client

        result = provider.recognize(
            Image.new("RGB", (40, 12), "white"),
            n_best=3,
            max_output_length=192,
        )

        self.assertEqual(
            [call["max_tokens"] for call in client.completions.calls],
            [768, 1600],
        )
        self.assertTrue(result.metadata["truncation_retry_attempted"])
        self.assertEqual(result.metadata["finish_reason_initial"], "length")
        self.assertEqual(result.metadata["finish_reason_final"], "stop")
        self.assertEqual(result.metadata["max_tokens_initial"], 768)
        self.assertEqual(result.metadata["max_tokens_final"], 1600)

    def test_normal_stop_does_not_retry(self):
        client = _FakeClient([_response(_payload("श्री"), "stop")])
        provider = QwenBedrockMantleHTRProvider()
        provider._client = lambda: client

        result = provider.recognize(
            Image.new("RGB", (40, 12), "white"),
            n_best=1,
            max_output_length=192,
        )

        self.assertEqual(
            [call["max_tokens"] for call in client.completions.calls],
            [768],
        )
        self.assertFalse(result.metadata["truncation_retry_attempted"])
        self.assertEqual(result.metadata["finish_reason_final"], "stop")

    def test_second_length_fails_truthfully(self):
        truncated = '{"script_guess":"Devanagari","candidates":['
        client = _FakeClient(
            [
                _response(truncated, "length"),
                _response(truncated, "length"),
            ]
        )
        provider = QwenBedrockMantleHTRProvider()
        provider._client = lambda: client

        with self.assertRaisesRegex(
            RuntimeError,
            "response remained truncated",
        ):
            provider.recognize(
                Image.new("RGB", (40, 12), "white"),
                n_best=3,
                max_output_length=192,
            )

        self.assertEqual(
            [call["max_tokens"] for call in client.completions.calls],
            [768, 1600],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
