#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo "Manuscript Intelligence — AWS Base Bootstrap"
echo "Repository: $REPO_ROOT"
echo "======================================================================"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: This bootstrap is for the EC2 Linux host, not macOS."
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "ERROR: Cannot identify Linux distribution."
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "ERROR: The validated AWS bootstrap target is Ubuntu 24.04 LTS."
  echo "Detected: ${PRETTY_NAME:-unknown}"
  exit 1
fi

echo "[1/5] Installing OS packages..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3.12 \
  python3.12-venv \
  python3.12-dev \
  build-essential \
  git \
  curl \
  unzip \
  ca-certificates \
  libglib2.0-0 \
  libgl1

if ! command -v aws >/dev/null 2>&1; then
  echo "[2/5] Installing AWS CLI v2..."
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64) AWS_ARCH="x86_64" ;;
    aarch64|arm64) AWS_ARCH="aarch64" ;;
    *)
      echo "ERROR: Unsupported architecture for AWS CLI v2: $ARCH"
      exit 1
      ;;
  esac

  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  curl -fsSL \
    "https://awscli.amazonaws.com/awscli-exe-linux-${AWS_ARCH}.zip" \
    -o "$TMP_DIR/awscliv2.zip"
  unzip -q "$TMP_DIR/awscliv2.zip" -d "$TMP_DIR"
  sudo "$TMP_DIR/aws/install" --update
else
  echo "[2/5] AWS CLI already installed: $(aws --version 2>&1)"
fi

echo "[3/5] Creating base Python 3.12 environment..."
if [[ ! -d .venv ]]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel

echo "[4/5] Installing base/AWS dependencies..."
.venv/bin/python -m pip install -r requirements-aws-base.txt

echo "[5/5] Verifying base imports..."
.venv/bin/python - <<'PY'
import cv2
import numpy
import fastapi
import uvicorn
import PIL
import boto3
import openai
from aws_bedrock_token_generator import provide_token

print("Python     : OK")
print("OpenCV     :", cv2.__version__)
print("NumPy      :", numpy.__version__)
print("FastAPI    :", fastapi.__version__)
print("Uvicorn    :", uvicorn.__version__)
print("Pillow     :", PIL.__version__)
print("boto3      :", boto3.__version__)
print("openai     :", openai.__version__)
print("Bedrock token generator import: OK")
PY

echo
echo "BASE BOOTSTRAP: PASS"
