from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import os
import cv2
import numpy as np

from core.io_utils import read_image_bgr, write_image
from core.artifact_store import ArtifactStore, RunMeta


@dataclass
class Layer0Output:
    raw_bgr: np.ndarray
    meta: Dict[str, Any]


def run_layer0_ingest(input_path: str, store: ArtifactStore, notes: str = "") -> Layer0Output:
    """
    L0: Ingestion & Context Capture

    What we achieve:
    - Read the raw manuscript image reliably.
    - Save a canonical raw copy inside the run folder.
    - Capture basic metadata (size, dtype, etc.).

    Why this layer exists:
    - In research demos, you must show the 'source of truth' (raw).
    - Ensures repeatability: same input, same output across runs.
    """

    raw = read_image_bgr(input_path)

    # Save canonical raw copy
    raw_path = store.path("L0", "raw.png")
    write_image(raw_path, raw)

    meta = {
        "input_path": os.path.abspath(input_path),
        "raw_shape_hwc": list(raw.shape),
        "raw_dtype": str(raw.dtype),
        "color_space": "BGR (OpenCV default)",
    }

    # Save run metadata
    run_meta = RunMeta(
        run_id=store.run_id,
        input_path=os.path.abspath(input_path),
        created_at_epoch=__import__("time").time(),
        notes=notes,
        params={"layer0": {"input_path": input_path}},
    )
    store.write_json("run_meta.json", run_meta.__dict__)
    store.write_json("L0/meta.json", meta)

    return Layer0Output(raw_bgr=raw, meta=meta)
