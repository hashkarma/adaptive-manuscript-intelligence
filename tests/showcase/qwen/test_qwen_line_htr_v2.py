import base64
import json
import time
from pathlib import Path

from openai import OpenAI
from aws_bedrock_token_generator import provide_token


AWS_REGION = "us-east-1"
BASE_URL = f"https://bedrock-mantle.{AWS_REGION}.api.aws/v1"
MODEL_ID = "qwen.qwen3-vl-235b-a22b-instruct"

IMAGE_PATH = Path("data/samples/golden_success/line1.png")
OUTPUT_PATH = Path("tests/fixtures/qwen/qwen_line1_htr_v2_response.txt")


PROMPT = """
You are performing STRICT VISUAL HTR on one cropped line from a historical
Devanagari manuscript.

Important:
- This is a visual transcription task, NOT a Sanskrit correction task.
- Read only what the image visibly contains.
- Do not repair spelling using grammar, dictionary knowledge, context, or likely phrases.
- Do not modernize orthography.
- Do not silently expand abbreviations.
- Do not translate.
- Do not infer missing words.
- If a glyph or akshara is unclear, write [?].
- Preserve visible danda marks such as । and ॥.
- Preserve apparent word boundaries only when visually supported.
- Your confidence must reflect visual certainty, not linguistic plausibility.

Work left-to-right and inspect the line at glyph/akshara level.

Return ONLY valid JSON in this exact structure:

{
  "script_guess": "",
  "visible_akshara_sequence": [
    "",
    ""
  ],
  "raw_visual_transcription": "",
  "overall_visual_confidence": 0.0,
  "uncertain_spans": [
    {
      "span": "",
      "alternatives": [""],
      "visual_reason": ""
    }
  ],
  "do_not_trust_without_review": [
    ""
  ]
}
"""


def image_as_data_url(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    token = provide_token()

    client = OpenAI(
        api_key=token,
        base_url=BASE_URL,
        timeout=180.0,
        max_retries=2,
    )

    print("Calling Qwen HTR v2 through Bedrock Mantle...")
    print(f"Model : {MODEL_ID}")
    print(f"Image : {IMAGE_PATH}")
    print(f"Bytes : {IMAGE_PATH.stat().st_size:,}")
    print()

    started = time.time()

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_as_data_url(IMAGE_PATH)
                        },
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=1200,
    )

    elapsed = time.time() - started
    result = response.choices[0].message.content or ""

    print("=" * 72)
    print("QWEN HTR V2 RESPONSE")
    print("=" * 72)
    print(result)
    print("=" * 72)
    print(f"Inference time: {elapsed:.2f} seconds")

    if getattr(response, "usage", None):
        print()
        print("USAGE")
        print(json.dumps(response.usage.model_dump(), indent=2))

    OUTPUT_PATH.write_text(result, encoding="utf-8")
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
