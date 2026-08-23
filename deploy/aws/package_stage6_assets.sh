#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-artifacts/deployment}"
OUT_FILE="${OUT_FILE:-$OUT_DIR/stage6-assets.tar.gz}"
S3_URI="${STAGE6_ASSET_S3_URI:-}"

mkdir -p "$OUT_DIR"

[[ -d models/vidyut-0.4.0 ]] || {
  echo "ERROR: models/vidyut-0.4.0 is missing."
  exit 1
}
[[ -f knowledge/stage6d_dcs/passages.jsonl ]] || {
  echo "ERROR: knowledge/stage6d_dcs/passages.jsonl is missing."
  exit 1
}

echo "Creating Stage-6 runtime asset archive..."
tar -czf "$OUT_FILE" \
  models/vidyut-0.4.0 \
  knowledge/stage6d_dcs

if command -v sha256sum >/dev/null 2>&1; then
  HASH="$(sha256sum "$OUT_FILE" | awk '{print $1}')"
else
  HASH="$(shasum -a 256 "$OUT_FILE" | awk '{print $1}')"
fi

printf '%s\n' "$HASH" > "${OUT_FILE}.sha256"

echo "Archive : $OUT_FILE"
echo "SHA256  : $HASH"
ls -lh "$OUT_FILE" "${OUT_FILE}.sha256"

if [[ -n "$S3_URI" ]]; then
  command -v aws >/dev/null 2>&1 || {
    echo "ERROR: AWS CLI not installed."
    exit 1
  }

  echo "Uploading to $S3_URI"
  aws s3 cp "$OUT_FILE" "$S3_URI"
  aws s3 cp "${OUT_FILE}.sha256" "${S3_URI}.sha256"
  echo "S3 upload: PASS"
else
  echo
  echo "STAGE6_ASSET_S3_URI is not set, so no upload was attempted."
  echo "Example:"
  echo "  export STAGE6_ASSET_S3_URI=s3://YOUR-BUCKET/manuscript-intelligence/stage6-assets.tar.gz"
fi
