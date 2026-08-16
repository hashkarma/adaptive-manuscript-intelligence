from __future__ import annotations

import platform
import sys


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("Architecture:", platform.machine())
    print("Platform:", platform.platform())

    try:
        import torch
    except ImportError:
        print("\nERROR: PyTorch is not installed.")
        raise SystemExit(1)

    print("PyTorch:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    print("CUDA available:", torch.cuda.is_available())

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Selected Stage 5 device:", device)

    try:
        import transformers
        print("Transformers:", transformers.__version__)
    except ImportError:
        print("Transformers: NOT INSTALLED")

    try:
        import indic_transliteration
        print("indic-transliteration: installed")
    except ImportError:
        print("indic-transliteration: NOT INSTALLED")


if __name__ == "__main__":
    main()