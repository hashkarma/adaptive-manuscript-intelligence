#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

S3_URI="${STAGE6_ASSET_S3_URI:-}"
FORCE="${FORCE_STAGE6_ASSETS:-0}"

if [[ -z "$S3_URI" ]]; then
  echo "ERROR: STAGE6_ASSET_S3_URI is required."
  echo "Example:"
  echo "  export STAGE6_ASSET_S3_URI=s3://YOUR-BUCKET/manuscript-intelligence/stage6-assets.tar.gz"
  exit 1
fi

if [[ "$FORCE" != "1" ]] && \
   [[ -d models/vidyut-0.4.0 || -d knowledge/stage6d_dcs ]]; then
  echo "ERROR: Stage-6 asset directories already exist."
  echo "Set FORCE_STAGE6_ASSETS=1 only if you intentionally want to overlay them."
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ARCHIVE="$TMP_DIR/stage6-assets.tar.gz"
EXPECTED_FILE="$TMP_DIR/stage6-assets.tar.gz.sha256"

echo "Downloading Stage-6 assets..."
aws s3 cp "$S3_URI" "$ARCHIVE"
aws s3 cp "${S3_URI}.sha256" "$EXPECTED_FILE"

EXPECTED="$(tr -d '[:space:]' < "$EXPECTED_FILE")"

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
else
  ACTUAL="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
fi

echo "Expected SHA256: $EXPECTED"
echo "Actual SHA256  : $ACTUAL"

if [[ "$EXPECTED" != "$ACTUAL" ]]; then
  echo "ERROR: Stage-6 asset checksum mismatch."
  exit 1
fi

echo "Checksum: PASS"
tar -xzf "$ARCHIVE" -C "$REPO_ROOT"

[[ -d models/vidyut-0.4.0 ]] || {
  echo "ERROR: Vidyut asset directory missing after extraction."
  exit 1
}
[[ -f knowledge/stage6d_dcs/passages.jsonl ]] || {
  echo "ERROR: DCS passages missing after extraction."
  exit 1
}

echo "STAGE 6 ASSET PROVISIONING: PASS"
