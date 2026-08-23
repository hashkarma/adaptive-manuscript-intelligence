from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_python(path: Path, code: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing interpreter: {path}"

    result = subprocess.run(
        [str(path), "-c", code],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode == 0, result.stdout.strip()


def check(label: str, ok: bool, detail: str = "") -> bool:
    state = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"{label:<42} {state}{suffix}")
    return ok


def main() -> None:
    os.chdir(ROOT)
    print("=" * 104)
    print("AWS MANUSCRIPT INTELLIGENCE RUNTIME VERIFICATION")
    print("=" * 104)

    results: list[bool] = []

    base = ROOT / ".venv/bin/python"
    a = ROOT / "venv-stage5/bin/python"
    b = ROOT / "venv-provider-b/bin/python"
    s6 = ROOT / "venv-stage6/bin/python"
    s6rag = ROOT / "venv-stage6-rag/bin/python"

    ok, detail = run_python(
        base,
        "import fastapi,cv2,numpy,PIL,boto3,openai;"
        "from aws_bedrock_token_generator import provide_token;"
        "print('base imports ok')",
    )
    results.append(check("Base environment", ok, detail))

    ok, detail = run_python(
        a,
        "import torch,transformers,numpy,PIL,indic_transliteration;"
        "print(torch.__version__, transformers.__version__)",
    )
    results.append(check("Stage 5 Provider A environment", ok, detail))

    ok, detail = run_python(
        b,
        "import torch,transformers,numpy,PIL,sentencepiece,indic_transliteration;"
        "print(torch.__version__, transformers.__version__)",
    )
    results.append(check("Stage 5 Provider B environment", ok, detail))

    ok, detail = run_python(
        s6,
        "import vidyut; print('vidyut import ok')",
    )
    results.append(check("Stage 6 Vidyut environment", ok, detail))

    ok, detail = run_python(
        s6rag,
        "import rapidfuzz; print(rapidfuzz.__version__)",
    )
    results.append(check("Stage 6 RAG environment", ok, detail))

    results.append(
        check(
            "Stage 6 Vidyut assets",
            (ROOT / "models/vidyut-0.4.0").is_dir(),
            "models/vidyut-0.4.0",
        )
    )
    results.append(
        check(
            "Stage 6 DCS corpus",
            (ROOT / "knowledge/stage6d_dcs/passages.jsonl").is_file(),
            "knowledge/stage6d_dcs/passages.jsonl",
        )
    )

    sys.path.insert(0, str(ROOT))
    try:
        from backend.main import app
        routes = {getattr(route, "path", "") for route in app.routes}
        expected = {
            "/upload",
            "/pipeline/stage1/restore/{run_id}",
            "/pipeline/stage2/damage/{run_id}",
            "/pipeline/stage3/layout/{run_id}",
            "/pipeline/stage4/segment/{run_id}",
            "/pipeline/stage5/htr/{run_id}",
            "/pipeline/stage6/run/{run_id}",
            "/showcase",
        }
        missing = sorted(expected - routes)
        results.append(
            check(
                "FastAPI Stage 0→6 routes",
                not missing,
                "missing=" + ",".join(missing) if missing else "all expected routes present",
            )
        )
    except Exception as exc:
        results.append(check("FastAPI Stage 0→6 routes", False, repr(exc)))

    if os.getenv("VERIFY_BEDROCK_TOKEN", "0") == "1":
        try:
            from aws_bedrock_token_generator import provide_token
            token = provide_token()
            results.append(check("Bedrock short-term token", bool(token), "generated"))
        except Exception as exc:
            results.append(check("Bedrock short-term token", False, repr(exc)))
    else:
        print(f"{'Bedrock short-term token':<42} SKIP — set VERIFY_BEDROCK_TOKEN=1")

    if os.getenv("VERIFY_BEDROCK_INFERENCE", "0") == "1":
        try:
            from aws_bedrock_token_generator import provide_token
            from openai import OpenAI

            region = os.getenv("AWS_REGION", "us-east-1")
            client = OpenAI(
                base_url=f"https://bedrock-mantle.{region}.api.aws/v1",
                api_key=provide_token(),
            )
            response = client.chat.completions.create(
                model="qwen.qwen3-vl-235b-a22b-instruct",
                messages=[{"role": "user", "content": "Reply with exactly: AWS_RUNTIME_OK"}],
                max_tokens=16,
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            results.append(
                check(
                    "Bedrock Qwen inference",
                    "AWS_RUNTIME_OK" in text,
                    text[:120],
                )
            )
        except Exception as exc:
            results.append(check("Bedrock Qwen inference", False, repr(exc)))
    else:
        print(f"{'Bedrock Qwen inference':<42} SKIP — set VERIFY_BEDROCK_INFERENCE=1")

    print("=" * 104)
    if all(results):
        print("AWS RUNTIME VERIFICATION: PASS")
    else:
        print("AWS RUNTIME VERIFICATION: FAIL")
        raise SystemExit(1)
    print("=" * 104)


if __name__ == "__main__":
    main()
