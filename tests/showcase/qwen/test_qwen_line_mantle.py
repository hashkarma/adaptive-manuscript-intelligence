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


PROMPT = """
You are examining one cropped handwritten line from a
historical Indian manuscript.

Perform a conservative transcription experiment.

Tasks:

1. Identify the most likely script.
2. Transcribe ONLY characters that are actually visible.
3. Preserve the original character sequence as much as possible.
4. Do NOT translate the text.
5. Do NOT modernize the spelling.
6. Do NOT silently reconstruct missing words.
7. Do NOT invent characters from linguistic context.
8. Wherever characters cannot be read confidently, use [?].
9. If multiple readings are possible, mention them under uncertainties.

Return ONLY valid JSON with this structure:

{
  "script_guess": "",
  "transcription": "",
  "confidence": 0.0,
  "uncertainties": [
    {
      "location": "",
      "possible_reading": "",
      "reason": ""
    }
  ],
  "visual_observations": ""
}
"""


def image_as_data_url(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    print("Generating short-term Bedrock token...")

    token = provide_token()

    print("Token generated: YES")
    print()

    client = OpenAI(
        api_key=token,
        base_url=BASE_URL,
        timeout=180.0,
        max_retries=2,
    )

    print("Calling Qwen through Bedrock Mantle...")
    print(f"Endpoint : {BASE_URL}")
    print(f"Model    : {MODEL_ID}")
    print(f"Image    : {IMAGE_PATH}")
    print(f"Bytes    : {IMAGE_PATH.stat().st_size:,}")
    print()

    start = time.time()

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
        temperature=0.1,
        max_tokens=800,
    )

    elapsed = time.time() - start

    result = response.choices[0].message.content

    print("=" * 70)
    print("QWEN RESPONSE")
    print("=" * 70)
    print(result)
    print("=" * 70)

    print()
    print(f"Inference time: {elapsed:.2f} seconds")

    if getattr(response, "usage", None):
        print()
        print("USAGE")
        print(
            json.dumps(
                response.usage.model_dump(),
                indent=2,
            )
        )

    output_path = Path(
        "tests/fixtures/qwen/qwen_line1_response.txt"
    )

    output_path.write_text(
        result or "",
        encoding="utf-8",
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
