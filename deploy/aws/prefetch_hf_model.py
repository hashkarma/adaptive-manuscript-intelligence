from __future__ import annotations

import os
import sys
from huggingface_hub import snapshot_download


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: prefetch_hf_model.py <huggingface_repo_id>")

    repo_id = sys.argv[1]
    print(f"Prefetching Hugging Face repository: {repo_id}")
    print(f"HF_HOME={os.getenv('HF_HOME', '<default>')}")

    path = snapshot_download(repo_id=repo_id)
    print(f"Cached: {path}")


if __name__ == "__main__":
    main()
