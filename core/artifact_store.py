from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class RunMeta:
    """
    Metadata saved for each run (very useful for demos, reproducibility, weekly progress).
    """
    run_id: str
    input_path: str
    created_at_epoch: float
    notes: str
    params: Dict[str, Any]


class ArtifactStore:
    """
    Simple artifact store (local folders).

    Why artifacts?
    - Professors like explainable intermediate outputs.
    - Every layer produces images/maps that you can show side-by-side.
    - Later, the same concept scales to S3/MinIO without changing the pipeline idea.
    """

    def __init__(self, base_dir: str, run_id: str):
        self.base_dir = base_dir
        self.run_id = run_id
        self.run_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

    def path(self, *parts: str) -> str:
        p = os.path.join(self.run_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def write_json(self, rel_path: str, data: dict) -> None:
        p = self.path(rel_path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def new_run_id(prefix: str = "run") -> str:
        # e.g., run_20260127_213012
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}"
