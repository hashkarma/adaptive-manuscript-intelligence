#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

[[ -x .venv/bin/python ]] || {
  echo "ERROR: .venv is missing. Run deploy/aws/bootstrap_base.sh."
  exit 1
}

echo "======================================================================"
echo "Adaptive Manuscript Intelligence Platform"
echo "Repository : $REPO_ROOT"
echo "AWS region : $AWS_REGION"
echo "Host/port  : $HOST:$PORT"
echo "Workers    : $WORKERS"
echo "======================================================================"

exec .venv/bin/python -m uvicorn backend.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS"
