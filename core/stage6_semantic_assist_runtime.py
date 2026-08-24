#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

CANDIDATE_STATUS = "AI CANDIDATE — SCHOLAR VALIDATION PENDING"
DEFAULT_BATCH_SIZE = 4


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


def split_batches(
    lines: list[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [
        lines[i : i + batch_size]
        for i in range(0, len(lines), batch_size)
    ]


def build_line_batch_prompt(lines: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['line_id']}: {row['observed_htr']}"
        for row in lines
    )

    return f"""
You are performing Stage 6 semantic assistance for a Sanskrit/Indic
manuscript research pipeline.

SCIENTIFIC RULES:
1. Stage-5 HTR is NOT ground truth.
2. Preserve every requested line_id exactly.
3. Produce a normalized Sanskrit/Devanagari candidate only when plausible.
4. Preserve uncertain observed spans instead of inventing certainty.
5. Do NOT invent verses, names, dates, places, titles, or technical claims.
6. Confidence is self-assessed and UNCALIBRATED; it is NOT accuracy.
7. No image is provided. Do NOT claim visual confirmation.
8. Output remains AI CANDIDATE — SCHOLAR VALIDATION PENDING.
9. Keep output extremely concise.
10. Return STRICT JSON ONLY, no Markdown.

Return exactly:
{{
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
          "alternatives": ["..."],
          "reason": "..."
        }}
      ],
      "edits": [
        {{
          "observed_span": "...",
          "proposed_span": "...",
          "rationale": "...",
          "evidence_basis": "linguistic_only|contextual_only|uncertain"
        }}
      ],
      "scholar_review_priority": "low|medium|high"
    }}
  ]
}}

Requested Stage-5 lines:
{payload}
""".strip()


def build_page_prompt(lines: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        (
            f"{row['line_id']} | observed={row.get('observed_htr','')} | "
            f"normalized={row.get('normalized_sanskrit_candidate','')} | "
            f"translation={row.get('english_translation_candidate','')}"
        )
        for row in lines
    )

    return f"""
You are producing only the PAGE-LEVEL scholar-assist synthesis for an
Indic manuscript.

RULES:
1. These are AI candidates, NOT ground truth.
2. Do NOT invent missing historical facts, names, dates, titles, places,
   authorship, or technical claims.
3. Preserve uncertainty.
4. Translation is NOT scholar verified.
5. Keep the response concise.
6. Return STRICT JSON ONLY.

Return exactly:
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
  "page_level": {{
    "normalized_page_candidate": "...",
    "english_translation_candidate": "...",
    "plain_english_summary": "...",
    "unresolved_items": ["..."],
    "translation_is_scholar_verified": false,
    "ground_truth_available": false
  }}
}}

Line-level candidates:
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


def call_json_with_retry(
    *,
    client: Any,
    model_id: str,
    prompt: str,
    primary_max_tokens: int,
    retry_max_tokens: int,
    label: str,
    out_dir: Path,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt_no, max_tokens in enumerate(
        (primary_max_tokens, retry_max_tokens),
        start=1,
    ):
        response = call_model(
            client=client,
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        finish_reason = response.choices[0].finish_reason
        raw = response.choices[0].message.content
        raw_text = raw if isinstance(raw, str) else str(raw)

        raw_path = out_dir / f"{label}_attempt_{attempt_no}_raw.txt"
        raw_path.write_text(raw_text + "\n", encoding="utf-8")

        record = {
            "label": label,
            "attempt": attempt_no,
            "max_tokens": max_tokens,
            "finish_reason": finish_reason,
            "raw_bytes": len(raw_text.encode("utf-8")),
        }

        try:
            parsed = extract_json_object(raw_text)
            record["parse_status"] = "ok"
            attempts.append(record)
            return parsed
        except Exception as exc:
            last_error = exc
            record["parse_status"] = "error"
            record["parse_error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(record)

    raise RuntimeError(
        f"{label} failed JSON parsing after retry: {last_error}"
    )


def run_semantic_assist(
    run_dir: Path,
    profile_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    profile = load_json(profile_path)
    stage6_cfg = profile.get("stage6", {}) or {}

    if not bool(stage6_cfg.get("qwen_semantic_assist_enabled")):
        raise RuntimeError(
            "Profile does not enable qwen_semantic_assist_enabled"
        )

    source_lines = read_live_stage5_lines(run_dir)
    batches = split_batches(source_lines, batch_size=batch_size)

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

    from openai import OpenAI
    from aws_bedrock_token_generator import provide_token

    client = OpenAI(
        base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
        api_key=provide_token(),
    )

    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(batches, start=1):
        label = f"line_batch_{batch_index:02d}"
        prompt = build_line_batch_prompt(batch)
        (out_dir / f"{label}_prompt.txt").write_text(
            prompt,
            encoding="utf-8",
        )

        parsed = call_json_with_retry(
            client=client,
            model_id=model_id,
            prompt=prompt,
            primary_max_tokens=3200,
            retry_max_tokens=5000,
            label=label,
            out_dir=out_dir,
            attempts=attempts,
        )

        rows = parsed.get("lines", [])
        if not isinstance(rows, list):
            raise RuntimeError(f"{label} returned no lines array")

        expected_ids = {row["line_id"] for row in batch}
        returned = {
            str(row.get("line_id"))
            for row in rows
            if isinstance(row, dict)
        }
        missing = expected_ids - returned
        if missing:
            raise RuntimeError(
                f"{label} missing line ids: {sorted(missing)}"
            )

        all_rows.extend(
            row for row in rows if isinstance(row, dict)
        )

    expected_order = {
        row["line_id"]: index
        for index, row in enumerate(source_lines)
    }
    deduped: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        line_id = str(row.get("line_id") or "")
        if line_id in expected_order:
            deduped[line_id] = row

    ordered_rows = [
        deduped[line_id]
        for line_id in sorted(
            deduped,
            key=lambda x: expected_order[x],
        )
    ]

    page_prompt = build_page_prompt(ordered_rows)
    (out_dir / "page_synthesis_prompt.txt").write_text(
        page_prompt,
        encoding="utf-8",
    )
    page_result = call_json_with_retry(
        client=client,
        model_id=model_id,
        prompt=page_prompt,
        primary_max_tokens=2200,
        retry_max_tokens=3500,
        label="page_synthesis",
        out_dir=out_dir,
        attempts=attempts,
    )

    result = {
        "document_assessment": page_result.get(
            "document_assessment",
            {},
        ),
        "lines": ordered_rows,
        "page_level": page_result.get("page_level", {}),
    }

    validation = validate_result(result, source_lines)
    runtime = time.perf_counter() - started

    write_json(
        out_dir / "stage6_semantic_candidate.json",
        result,
    )

    audit = {
        "artifact_version": "1.1.0-live-stage6-semantic-assist-batched",
        "profile_id": profile.get("profile_id"),
        "model_id": model_id,
        "runtime_seconds": round(runtime, 3),
        "execution_strategy": (
            "batched_line_semantic_assist_plus_page_synthesis"
        ),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "input_transcription": "L5/page_transcription.json",
        "input_line_count": len(source_lines),
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
    if validation["extra_line_ids"]:
        raise RuntimeError(
            "Semantic assist returned unexpected line ids: "
            + ",".join(validation["extra_line_ids"])
        )

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live Stage-6 batched Qwen semantic-assist candidate runtime."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    args = parser.parse_args()

    audit = run_semantic_assist(
        run_dir=Path(args.run_dir).resolve(),
        profile_path=Path(args.profile).resolve(),
        batch_size=args.batch_size,
    )

    print("STAGE6_SEMANTIC_ASSIST=PASS")
    print(
        json.dumps(
            {
                "runtime_seconds": audit.get("runtime_seconds"),
                "execution_strategy": audit.get(
                    "execution_strategy"
                ),
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
