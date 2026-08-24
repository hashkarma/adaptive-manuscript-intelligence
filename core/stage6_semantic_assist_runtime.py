#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

CANDIDATE_STATUS = "AI CANDIDATE — SCHOLAR VALIDATION PENDING"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No complete JSON object found in model response")

    data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Model JSON payload is not an object")
    return data


def read_live_stage5_lines(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "L5" / "page_transcription.json"
    data = load_json(path)
    rows = data.get("lines", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No Stage-5 lines found in {path}")

    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        text = row.get("devanagari_text")
        if not isinstance(text, str) or not text.strip():
            text = row.get("raw_text")
        if not isinstance(text, str) or not text.strip():
            continue

        out.append(
            {
                "line_id": str(row.get("line_id")),
                "reading_order": int(
                    row.get("reading_order") or len(out) + 1
                ),
                "observed_htr": text.strip(),
                "review_required": bool(
                    row.get("review_required", False)
                ),
            }
        )

    out.sort(key=lambda x: (x["reading_order"], x["line_id"]))
    if not out:
        raise ValueError(f"No usable Stage-5 text found in {path}")
    return out


def build_prompt(lines: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['line_id']}: {row['observed_htr']}"
        for row in lines
    )

    return f"""
You are performing Stage 6 semantic assistance for an Indic manuscript
research pipeline.

RESEARCH RULES:
1. The Stage-5 HTR below is NOT ground truth.
2. Preserve every line_id exactly.
3. Produce a useful normalized Sanskrit/Devanagari candidate when
   linguistically plausible.
4. If a span cannot be justified, preserve the observed reading and mark
   it uncertain.
5. Do NOT invent missing verses, names, dates, places, titles, or technical
   claims just to make the text grammatical.
6. Distinguish observed HTR from proposed normalization and English
   translation.
7. Every correction must be auditable in "edits".
8. Allowed evidence_basis values are:
   "linguistic_only", "contextual_only", "uncertain".
9. Confidence values are self-assessed and UNCALIBRATED; never call them
   accuracy.
10. No page image is provided in this Stage-6 call. Visual recognition was
    already performed in Stage 5. Do NOT claim visual confirmation for any
    Stage-6 correction.
11. The goal is a SCHOLAR-ASSIST PACKET: useful candidates plus explicit
    uncertainty.
12. Keep explanations concise so the JSON remains within the model response
    budget.
13. Return STRICT JSON ONLY. Do not wrap it in Markdown.

Return this schema:
{{
  "document_assessment": {{
    "script": "Devanagari",
    "language_guess": "...",
    "work_or_title_guess": "... or null",
    "genre_guess": "... or null",
    "overall_notes": "...",
    "candidate_output_status":
      "AI_CANDIDATE_SCHOLAR_VALIDATION_PENDING"
  }},
  "lines": [
    {{
      "line_id": "line_001",
      "observed_htr": "...",
      "normalized_sanskrit_candidate": "...",
      "english_translation_candidate": "...",
      "candidate_confidence_uncalibrated": 0.0,
      "uncertain_spans": [
        {{
          "span": "...",
          "alternatives": ["...", "..."],
          "reason": "..."
        }}
      ],
      "edits": [
        {{
          "observed_span": "...",
          "proposed_span": "...",
          "rationale": "...",
          "evidence_basis": "linguistic_only"
        }}
      ],
      "scholar_review_priority": "low|medium|high"
    }}
  ],
  "page_level": {{
    "normalized_page_candidate": "...",
    "english_translation_candidate": "...",
    "plain_english_summary": "...",
    "unresolved_items": ["..."],
    "translation_is_scholar_verified": false,
    "ground_truth_available": false
  }}
}}

Observed Stage-5 HTR lines:
{payload}
""".strip()


def validate_result(
    result: dict[str, Any],
    expected_lines: list[dict[str, Any]],
) -> dict[str, Any]:
    problems: list[str] = []
    rows = result.get("lines")

    if not isinstance(rows, list):
        problems.append("RESULT_LINES_NOT_LIST")
        rows = []

    expected_ids = [row["line_id"] for row in expected_lines]
    returned_ids = [
        str(row.get("line_id"))
        for row in rows
        if isinstance(row, dict) and row.get("line_id") is not None
    ]

    missing = [x for x in expected_ids if x not in returned_ids]
    extra = [x for x in returned_ids if x not in expected_ids]

    if missing:
        problems.append(f"MISSING_LINE_IDS:{','.join(missing)}")
    if extra:
        problems.append(f"EXTRA_LINE_IDS:{','.join(extra)}")

    normalized = 0
    translated = 0
    high_review = 0

    for row in rows:
        if not isinstance(row, dict):
            continue

        norm = row.get("normalized_sanskrit_candidate")
        trans = row.get("english_translation_candidate")
        priority = str(row.get("scholar_review_priority", "")).lower()

        if isinstance(norm, str) and norm.strip():
            normalized += 1
        if isinstance(trans, str) and trans.strip():
            translated += 1
        if priority == "high":
            high_review += 1

    page = result.get("page_level", {})
    page_translation = (
        page.get("english_translation_candidate")
        if isinstance(page, dict)
        else None
    )

    return {
        "status": "ok" if not problems else "warning",
        "problems": problems,
        "expected_lines": len(expected_ids),
        "returned_lines": len(returned_ids),
        "missing_line_ids": missing,
        "extra_line_ids": extra,
        "normalized_candidate_lines": normalized,
        "translated_candidate_lines": translated,
        "high_priority_review_lines": high_review,
        "page_translation_available": (
            isinstance(page_translation, str)
            and bool(page_translation.strip())
        ),
    }


def call_model(
    client: Any,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> Any:
    return client.chat.completions.create(
        model=model_id,
        temperature=0.1,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    )


def merge_missing_lines(
    base_result: dict[str, Any],
    continuation: dict[str, Any],
    expected_lines: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_order = {
        row["line_id"]: index
        for index, row in enumerate(expected_lines)
    }

    merged: dict[str, dict[str, Any]] = {}
    for source in (
        base_result.get("lines", []),
        continuation.get("lines", []),
    ):
        if not isinstance(source, list):
            continue
        for row in source:
            if not isinstance(row, dict):
                continue
            line_id = str(row.get("line_id") or "")
            if line_id in expected_order:
                merged[line_id] = row

    result = dict(base_result)
    result["lines"] = [
        merged[line_id]
        for line_id in sorted(
            merged,
            key=lambda x: expected_order[x],
        )
    ]

    if not isinstance(result.get("page_level"), dict):
        if isinstance(continuation.get("page_level"), dict):
            result["page_level"] = continuation["page_level"]

    if not isinstance(result.get("document_assessment"), dict):
        if isinstance(
            continuation.get("document_assessment"),
            dict,
        ):
            result["document_assessment"] = continuation[
                "document_assessment"
            ]

    return result


def run_semantic_assist(
    run_dir: Path,
    profile_path: Path,
    max_tokens: int = 7000,
) -> dict[str, Any]:
    profile = load_json(profile_path)
    stage6_cfg = profile.get("stage6", {}) or {}

    if not bool(stage6_cfg.get("qwen_semantic_assist_enabled")):
        raise RuntimeError(
            "Profile does not enable qwen_semantic_assist_enabled"
        )

    lines = read_live_stage5_lines(run_dir)
    prompt = build_prompt(lines)

    model_id = (
        (profile.get("stage5", {}) or {})
        .get("bedrock", {})
        .get(
            "model_id",
            "qwen.qwen3-vl-235b-a22b-instruct",
        )
    )

    out_dir = run_dir / "L6_semantic_assist"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage6_semantic_prompt.txt").write_text(
        prompt,
        encoding="utf-8",
    )

    from openai import OpenAI
    from aws_bedrock_token_generator import provide_token

    client = OpenAI(
        base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
        api_key=provide_token(),
    )

    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []

    response = call_model(
        client=client,
        model_id=model_id,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    finish_reason = response.choices[0].finish_reason
    raw = response.choices[0].message.content
    raw_text = raw if isinstance(raw, str) else str(raw)
    attempts.append(
        {
            "max_tokens": max_tokens,
            "finish_reason": finish_reason,
        }
    )

    if finish_reason == "length" and max_tokens < 8000:
        retry_prompt = (
            prompt
            + "\n\nIMPORTANT: The previous response exceeded the output "
              "budget. Return the same schema for all lines, but make notes, "
              "edits, and uncertainty explanations extremely concise."
        )
        (out_dir / "stage6_semantic_prompt_retry.txt").write_text(
            retry_prompt,
            encoding="utf-8",
        )
        response = call_model(
            client=client,
            model_id=model_id,
            prompt=retry_prompt,
            max_tokens=8000,
        )
        finish_reason = response.choices[0].finish_reason
        raw = response.choices[0].message.content
        raw_text = raw if isinstance(raw, str) else str(raw)
        attempts.append(
            {
                "max_tokens": 8000,
                "finish_reason": finish_reason,
            }
        )

    raw_path = out_dir / "stage6_semantic_raw_response.txt"
    raw_path.write_text(raw_text + "\n", encoding="utf-8")

    result = extract_json_object(raw_text)
    validation = validate_result(result, lines)

    if validation["missing_line_ids"]:
        missing_ids = set(validation["missing_line_ids"])
        missing_lines = [
            row for row in lines if row["line_id"] in missing_ids
        ]
        continuation_prompt = (
            build_prompt(missing_lines)
            + "\n\nCONTINUATION RULE: Return ONLY the requested missing "
              "line_ids in the lines array. Keep the page_level section "
              "concise."
        )
        (
            out_dir / "stage6_semantic_continuation_prompt.txt"
        ).write_text(
            continuation_prompt,
            encoding="utf-8",
        )

        continuation_response = call_model(
            client=client,
            model_id=model_id,
            prompt=continuation_prompt,
            max_tokens=5000,
        )
        continuation_finish = (
            continuation_response.choices[0].finish_reason
        )
        continuation_raw = (
            continuation_response.choices[0].message.content
        )
        continuation_text = (
            continuation_raw
            if isinstance(continuation_raw, str)
            else str(continuation_raw)
        )
        (
            out_dir / "stage6_semantic_continuation_raw_response.txt"
        ).write_text(
            continuation_text + "\n",
            encoding="utf-8",
        )
        attempts.append(
            {
                "kind": "missing_line_continuation",
                "max_tokens": 5000,
                "finish_reason": continuation_finish,
                "requested_line_ids": sorted(missing_ids),
            }
        )

        continuation = extract_json_object(continuation_text)
        result = merge_missing_lines(
            base_result=result,
            continuation=continuation,
            expected_lines=lines,
        )
        validation = validate_result(result, lines)

    runtime = time.perf_counter() - started

    write_json(
        out_dir / "stage6_semantic_candidate.json",
        result,
    )

    audit = {
        "artifact_version": "1.0.0-live-stage6-semantic-assist",
        "profile_id": profile.get("profile_id"),
        "model_id": model_id,
        "runtime_seconds": round(runtime, 3),
        "input_transcription": "L5/page_transcription.json",
        "input_line_count": len(lines),
        "attempts": attempts,
        "validation": validation,
        "scientific_status": {
            "candidate_output_status": CANDIDATE_STATUS,
            "output_is_ai_candidate": True,
            "scholar_verified": False,
            "ground_truth_available": False,
            "confidence_values_are_calibrated": False,
            "cer": None,
            "wer": None,
            "stage5_was_visual_htr": True,
            "stage6_visual_confirmation_used": False,
            "changes_S_H_T": False,
        },
    }
    write_json(
        out_dir / "stage6_semantic_audit.json",
        audit,
    )

    if validation["missing_line_ids"]:
        raise RuntimeError(
            "Semantic assist did not return every expected line: "
            + ",".join(validation["missing_line_ids"])
        )

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Stage-6 Qwen semantic-assist candidate runtime."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--max-tokens", type=int, default=7000)
    args = parser.parse_args()

    audit = run_semantic_assist(
        run_dir=Path(args.run_dir).resolve(),
        profile_path=Path(args.profile).resolve(),
        max_tokens=args.max_tokens,
    )

    print("STAGE6_SEMANTIC_ASSIST=PASS")
    print(
        json.dumps(
            {
                "runtime_seconds": audit.get("runtime_seconds"),
                "validation": audit.get("validation"),
                "scientific_status": audit.get(
                    "scientific_status"
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
