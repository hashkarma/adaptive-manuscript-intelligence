#!/usr/bin/env bash
set -u

echo "============================================================"
echo " Git pre-commit audit - Manuscript Intelligence Platform"
echo "============================================================"
echo

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: This directory is not inside a Git repository."
  exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
echo "Repository root:"
echo "  $ROOT"
echo

echo "Current branch:"
git branch --show-current || true
echo

echo "Configured remotes:"
git remote -v || true
echo

ORIGIN="$(git remote get-url origin 2>/dev/null || true)"
if [ -n "$ORIGIN" ]; then
  if printf '%s\n' "$ORIGIN" | grep -Eiq 'github\.com[:/]hashkarma/'; then
    echo "HashKarma remote check: OK"
    echo "  origin = $ORIGIN"
  else
    echo "HashKarma remote check: REVIEW"
    echo "  origin = $ORIGIN"
    echo "  This does not appear to be a github.com/hashkarma/... remote."
  fi
else
  echo "HashKarma remote check: REVIEW"
  echo "  No origin remote is configured."
fi
echo

echo "------------------------------------------------------------"
echo "Working tree status"
echo "------------------------------------------------------------"
git status --short
echo

echo "------------------------------------------------------------"
echo "Files larger than 25 MB anywhere under the repo"
echo "(Ignored files may still appear here; this is intentional.)"
echo "------------------------------------------------------------"
find "$ROOT" \
  -type f \
  -size +25M \
  -not -path "$ROOT/.git/*" \
  -print 2>/dev/null | sed "s#^$ROOT/##" | sort
echo

echo "------------------------------------------------------------"
echo "Potentially large/generated paths that are ALREADY tracked"
echo "These need git rm --cached if they should be excluded."
echo "------------------------------------------------------------"
git ls-files | grep -E \
'^(artifacts/|models/|data/raw/|knowledge/stage6d_dcs/|venv/|venv-[^/]+/|\.venv/|runs/|logs/)' \
|| echo "None detected."
echo

echo "------------------------------------------------------------"
echo "Tracked ML/model/data files that deserve review"
echo "------------------------------------------------------------"
git ls-files | grep -Ei \
'\.(safetensors|ckpt|pt|pth|onnx|npy|npz|faiss|parquet|arrow|h5|hdf5|tflite)$' \
|| echo "None detected."
echo

echo "============================================================"
echo "Audit complete."
echo "No files were added, removed, committed, or pushed."
echo "============================================================"
