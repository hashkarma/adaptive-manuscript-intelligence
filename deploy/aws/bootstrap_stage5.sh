#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
PYTORCH_CPU_INDEX_URL="${PYTORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
PREFETCH_MODELS="${PREFETCH_MODELS:-1}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME

echo "======================================================================"
echo "Manuscript Intelligence — Stage 5 Linux Bootstrap"
echo "HF_HOME: $HF_HOME"
echo "======================================================================"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: $PYTHON_BIN not found. Run deploy/aws/bootstrap_base.sh first."
  exit 1
}

create_env() {
  local env_name="$1"
  if [[ ! -d "$env_name" ]]; then
    "$PYTHON_BIN" -m venv "$env_name"
  fi
  "$env_name/bin/python" -m pip install --upgrade pip setuptools wheel
}

echo "[1/6] Creating Provider A environment: venv-stage5"
create_env "venv-stage5"

echo "[2/6] Installing Provider A CPU PyTorch..."
venv-stage5/bin/python -m pip install \
  --index-url "$PYTORCH_CPU_INDEX_URL" \
  "torch==2.13.0" "torchvision==0.28.0"
venv-stage5/bin/python -m pip install -r requirements-stage5-a.txt

echo "[3/6] Creating Provider B environment: venv-provider-b"
create_env "venv-provider-b"

echo "[4/6] Installing Provider B CPU PyTorch..."
venv-provider-b/bin/python -m pip install \
  --index-url "$PYTORCH_CPU_INDEX_URL" \
  "torch==2.13.0" "torchvision==0.28.0"
venv-provider-b/bin/python -m pip install -r requirements-stage5-b.txt

echo "[5/6] Verifying Provider A/B imports..."
venv-stage5/bin/python - <<'PY'
import torch, transformers, numpy, PIL, indic_transliteration
print("Provider A torch       :", torch.__version__)
print("Provider A transformers:", transformers.__version__)
print("Provider A numpy       :", numpy.__version__)
print("Provider A device      : CPU")
PY

venv-provider-b/bin/python - <<'PY'
import torch, transformers, numpy, PIL, sentencepiece, indic_transliteration
print("Provider B torch       :", torch.__version__)
print("Provider B transformers:", transformers.__version__)
print("Provider B numpy       :", numpy.__version__)
print("Provider B device      : CPU")
PY

echo "[6/6] Prefetching Hugging Face models..."
if [[ "$PREFETCH_MODELS" == "1" ]]; then
  venv-stage5/bin/python deploy/aws/prefetch_hf_model.py \
    "Piyush3142/trocr-sanskrit-ocr"

  venv-provider-b/bin/python deploy/aws/prefetch_hf_model.py \
    "yzk/trocr-large-printed-vedic"

  venv-provider-b/bin/python deploy/aws/prefetch_hf_model.py \
    "microsoft/trocr-large-printed"
else
  echo "PREFETCH_MODELS=$PREFETCH_MODELS — model prefetch skipped."
fi

echo
echo "STAGE 5 BOOTSTRAP: PASS"
