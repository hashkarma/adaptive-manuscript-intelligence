#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

echo "======================================================================"
echo "Manuscript Intelligence — Stage 6 Linux Bootstrap"
echo "======================================================================"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: $PYTHON_BIN not found. Run deploy/aws/bootstrap_base.sh first."
  exit 1
}

create_env() {
  local env_name="$1"
  local requirements_file="$2"

  if [[ ! -d "$env_name" ]]; then
    "$PYTHON_BIN" -m venv "$env_name"
  fi

  "$env_name/bin/python" -m pip install --upgrade pip setuptools wheel
  "$env_name/bin/python" -m pip install -r "$requirements_file"
}

echo "[1/4] Creating deterministic Stage-6/Vidyut environment..."
create_env "venv-stage6" "requirements-stage6.txt"

echo "[2/4] Creating Stage-6/RAG environment..."
create_env "venv-stage6-rag" "requirements-stage6-rag.txt"

echo "[3/4] Verifying isolated Stage-6 dependencies..."
venv-stage6/bin/python - <<'PY'
import vidyut
print("Vidyut:", getattr(vidyut, "__version__", "imported"))
PY

venv-stage6-rag/bin/python - <<'PY'
import rapidfuzz
print("RapidFuzz:", rapidfuzz.__version__)
PY

echo "[4/4] Checking Stage-6 research assets..."
if [[ -d models/vidyut-0.4.0 && -f knowledge/stage6d_dcs/passages.jsonl ]]; then
  echo "Stage-6 assets are already present."
else
  echo "Stage-6 assets are not present yet."
  echo "Next run: deploy/aws/bootstrap_stage6_assets.sh after configuring STAGE6_ASSET_S3_URI."
fi

echo
echo "STAGE 6 PYTHON BOOTSTRAP: PASS"
