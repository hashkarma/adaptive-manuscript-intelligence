from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from PIL import Image
from openai import OpenAI
from aws_bedrock_token_generator import provide_token

from layers.htr_providers import HTRProvider, ProviderHypothesis, ProviderResult


DEFAULT_QWEN_BEDROCK_MODEL_ID = "qwen.qwen3-vl-235b-a22b-instruct"
DEFAULT_QWEN_BEDROCK_REGION = "us-east-1"


STRICT_HTR_PROMPT = """
You are performing STRICT VISUAL HTR on one cropped line from a historical
Devanagari manuscript.

This is visual transcription, not Sanskrit correction.

Rules:
- Read only glyphs/aksharas visibly present in the image.
- Do not repair spelling using grammar, dictionary knowledge, context, or a
  likely Sanskrit phrase.
- Do not modernize orthography.
- Do not translate.
- Do not expand abbreviations silently.
- Do not infer missing words.
- Preserve danda marks such as । and ॥ when visible.
- Preserve word boundaries only when visually supported.
- Mark unreadable spans with [?].
- Confidence must reflect visual certainty, not linguistic plausibility.

Return ONLY valid JSON in this exact structure:

{
  "script_guess": "Devanagari",
  "candidates": [
    {
      "text": "",
      "confidence": 0.0
    }
  ],
  "uncertainties": [
    {
      "span": "",
      "alternatives": [""],
      "visual_reason": ""
    }
  ],
  "visual_observations": ""
}

Provide up to __N_BEST__ genuinely distinct visual candidates. Do not invent
alternatives merely to fill the list. The first candidate must be your best
strict visual transcription.
"""


def _strip_json_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _clip01(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default



# PROVIDER_C_STRUCTURED_RESPONSE_PARSER_V2
_PROTOCOL_MARKERS = (
    '"script_guess"',
    '"candidates"',
    '"confidence"',
    '"uncertainties"',
)


def _looks_like_protocol_payload(text: str) -> bool:
    value = _strip_json_fence(text)
    if not value:
        return False

    marker_hits = sum(
        1
        for marker in _PROTOCOL_MARKERS
        if marker in value
    )

    return bool(
        value.lstrip().startswith("{")
        and marker_hits >= 2
    )


def _parse_json_dict_with_recovery(
    text: str,
) -> tuple[Dict[str, Any], List[str]]:
    value = _strip_json_fence(text)
    actions: List[str] = []

    if not value:
        raise ValueError("Qwen response was empty.")

    try:
        parsed: Any = json.loads(value)
    except Exception as strict_exc:
        start = value.find("{")
        end = value.rfind("}")

        if start < 0 or end <= start:
            raise ValueError(
                "Qwen response was not valid structured JSON."
            ) from strict_exc

        bounded = value[start : end + 1]

        try:
            parsed = json.loads(bounded)
        except Exception as bounded_exc:
            raise ValueError(
                "Qwen JSON-like response could not be parsed safely."
            ) from bounded_exc

        actions.append("bounded_json_object_extracted")

    for _ in range(2):
        if not isinstance(parsed, str):
            break

        try:
            parsed = json.loads(
                _strip_json_fence(parsed)
            )
        except Exception as nested_exc:
            raise ValueError(
                "Qwen JSON string envelope could not be decoded safely."
            ) from nested_exc

        actions.append(
            "json_string_envelope_unwrapped"
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            "Qwen structured response must decode to a JSON object."
        )

    return parsed, actions


def _parse_qwen_structured_response(
    raw_response: str,
) -> tuple[Dict[str, Any], List[str]]:
    parsed, actions = _parse_json_dict_with_recovery(
        raw_response
    )

    for depth in range(2):
        rows = parsed.get("candidates")

        if not isinstance(rows, list):
            break

        nested_payload = None

        for row in rows:
            if not isinstance(row, dict):
                continue

            candidate_text = str(
                row.get("text", "")
                or ""
            ).strip()

            if not _looks_like_protocol_payload(
                candidate_text
            ):
                continue

            try:
                possible, nested_actions = (
                    _parse_json_dict_with_recovery(
                        candidate_text
                    )
                )
            except Exception:
                continue

            if isinstance(
                possible.get("candidates"),
                list,
            ):
                nested_payload = possible
                actions.extend(nested_actions)
                actions.append(
                    "nested_protocol_payload_unwrapped_"
                    f"depth_{depth + 1}"
                )
                break

        if nested_payload is None:
            break

        parsed = nested_payload

    if not isinstance(
        parsed.get("candidates"),
        list,
    ):
        raise ValueError(
            "Qwen structured response has no candidates list."
        )

    return (
        parsed,
        list(dict.fromkeys(actions)),
    )


class QwenBedrockMantleHTRProvider(HTRProvider):
    provider_id = "qwen_bedrock_mantle"
    output_script = "devanagari"

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_BEDROCK_MODEL_ID,
        *,
        region: Optional[str] = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        self._model_id = model_id
        self._region = (
            region
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or DEFAULT_QWEN_BEDROCK_REGION
        )
        self._base_url = f"https://bedrock-mantle.{self._region}.api.aws/v1"
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def processor_id(self) -> str:
        return "bedrock-mantle-openai-compatible"

    @property
    def device(self) -> str:
        return "aws-bedrock"

    @property
    def device_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_id,
            "execution": "managed_remote_inference",
            "region": self._region,
            "endpoint": self._base_url,
            "local_gpu_required": False,
        }

    @staticmethod
    def _image_as_data_url(image: Image.Image) -> str:
        rgb = image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="PNG", optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _client(self) -> OpenAI:
        # PROVIDER_C_REGION_PROPAGATION_V1
        # Propagate the already-resolved region to the token generator.
        # Credentials still come from the normal AWS provider chain.
        os.environ.setdefault("AWS_REGION", self._region)
        os.environ.setdefault("AWS_DEFAULT_REGION", self._region)
        token = provide_token()
        return OpenAI(
            api_key=token,
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            max_retries=self._max_retries,
        )

    def recognize(
        self,
        image: Image.Image,
        *,
        num_beams: int = 4,
        n_best: int = 3,
        max_output_length: int = 192,
    ) -> ProviderResult:
        started = time.perf_counter()

        requested_n_best = max(1, min(int(n_best), 5))
        prompt = STRICT_HTR_PROMPT.replace("__N_BEST__", str(requested_n_best))
        # PROVIDER_C_TRUNCATION_RETRY_V1
        # Preserve the existing economical initial budget. If Mantle
        # explicitly reports finish_reason="length", retry the same visual
        # request once at the already-supported 1600-token ceiling.
        # Truncated/raw protocol text is never promoted to transcription.
        initial_max_tokens = max(
            600,
            min(1600, int(max_output_length) * 4),
        )
        max_tokens = initial_max_tokens

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_as_data_url(image)
                        },
                    },
                ],
            }
        ]

        client = self._client()
        response = client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )

        initial_finish_reason = (
            getattr(response.choices[0], "finish_reason", None)
            if response.choices
            else None
        )
        truncation_retry_attempted = False

        if (
            initial_finish_reason == "length"
            and initial_max_tokens < 1600
        ):
            truncation_retry_attempted = True
            max_tokens = 1600
            response = client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )

        final_finish_reason = (
            getattr(response.choices[0], "finish_reason", None)
            if response.choices
            else None
        )
        runtime_ms = (time.perf_counter() - started) * 1000.0

        if final_finish_reason == "length":
            raise RuntimeError(
                "Qwen Bedrock provider response remained truncated "
                "(finish_reason=length, "
                f"initial_max_tokens={initial_max_tokens}, "
                f"final_max_tokens={max_tokens}). "
                "Raw truncated protocol text was rejected as transcription."
            )

        raw_response = (
            response.choices[0].message.content
            if response.choices
            else ""
        ) or ""

        # PROVIDER_C_NO_RAW_PROTOCOL_FALLBACK_V2
        # Protocol JSON must never enter the manuscript transcription path.
        # Recover structured output narrowly; otherwise fail this line
        # truthfully so Layer 5 can mark it for review/error.
        try:
            (
                parsed,
                json_recovery_actions,
            ) = _parse_qwen_structured_response(
                raw_response
            )
            parse_error: Optional[str] = None
        except Exception as exc:
            parse_error = (
                f"{type(exc).__name__}: {exc}"
            )
            raise RuntimeError(
                "Qwen Bedrock provider returned malformed "
                "structured HTR output. Raw protocol text "
                "was rejected as transcription. "
                f"{parse_error}"
            ) from exc

        candidate_rows = parsed.get("candidates")
        if not isinstance(candidate_rows, list):
            candidate_rows = []

        seen = set()
        candidates: List[Dict[str, Any]] = []

        for row in candidate_rows:
            if not isinstance(row, dict):
                continue

            text = " ".join(str(row.get("text", "")).strip().split())
            if _looks_like_protocol_payload(
                text
            ):
                continue

            if not text or text in seen:
                continue

            seen.add(text)
            candidates.append(
                {
                    "text": text,
                    "confidence": _clip01(row.get("confidence"), default=0.5),
                }
            )

            if len(candidates) >= requested_n_best:
                break

        if not candidates:
            raise RuntimeError(
                "Qwen Bedrock provider returned no usable transcription."
            )

        score_sum = sum(max(c["confidence"], 1e-6) for c in candidates)

        hypotheses: List[ProviderHypothesis] = []
        for rank, candidate in enumerate(candidates, start=1):
            hypotheses.append(
                ProviderHypothesis(
                    rank=rank,
                    raw_text=candidate["text"],
                    relative_score=(
                        max(candidate["confidence"], 1e-6) / score_sum
                    ),
                    sequence_score=None,
                )
            )

        usage = None
        if getattr(response, "usage", None) is not None:
            try:
                usage = response.usage.model_dump()
            except Exception:
                usage = str(response.usage)

        metadata: Dict[str, Any] = {
            "provider_id": self.provider_id,
            "model_id": self._model_id,
            "region": self._region,
            "endpoint": "bedrock-mantle",
            "api": "chat.completions",
            "requested_n_best": requested_n_best,
            "returned_n_best": len(hypotheses),
            "num_beams_ignored": int(num_beams),
            "model_self_reported_confidences": [
                c["confidence"] for c in candidates
            ],
            "script_guess": parsed.get("script_guess"),
            "uncertainties": parsed.get("uncertainties", []),
            "visual_observations": parsed.get("visual_observations", ""),
            "json_parse_error": parse_error,
            "json_recovery_actions": json_recovery_actions,
            "finish_reason_initial": initial_finish_reason,
            "finish_reason_final": final_finish_reason,
            "max_tokens_initial": initial_max_tokens,
            "max_tokens_final": max_tokens,
            "truncation_retry_attempted": truncation_retry_attempted,
            "truncation_retry_policy": (
                "retry_once_at_1600_on_finish_reason_length"
            ),
            "usage": usage,
        }

        return ProviderResult(
            hypotheses=hypotheses,
            runtime_ms=runtime_ms,
            device_used="aws-bedrock",
            metadata=metadata,
        )
