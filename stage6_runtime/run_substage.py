from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

from core.artifact_store import ArtifactStore


RUNNER_VERSION = "0.1.0-stage6-isolated-substage-runner"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _dispatch(
    substage: str,
    store: ArtifactStore,
    *,
    vidyut_data: str,
    corpus: str,
) -> Dict[str, Any]:
    """
    Import only the selected substage.

    This is important because venv-stage6 intentionally does not need
    RapidFuzz, and venv-stage6-rag intentionally does not need Vidyut.
    """
    if substage == "6a":
        from layers.layer6_semantic import run_layer6_htr_evidence_parser

        return run_layer6_htr_evidence_parser(store)

    if substage == "6b":
        from layers.layer6b_reconstruct import (
            run_layer6b_candidate_reconstruction,
        )

        return run_layer6b_candidate_reconstruction(
            store,
            vidyut_data_root=vidyut_data,
        )

    if substage == "6c":
        from layers.layer6c_morphology import (
            run_layer6c_morphology_validation,
        )

        return run_layer6c_morphology_validation(
            store,
            vidyut_data_root=vidyut_data,
        )

    if substage == "6d":
        from layers.layer6d_context_rag import (
            run_layer6d_contextual_retrieval,
        )

        return run_layer6d_contextual_retrieval(
            store,
            corpus_path=corpus,
        )

    if substage == "6d2":
        from layers.layer6d2_noisy_retrieval import (
            run_layer6d2_noisy_surface_retrieval,
        )

        return run_layer6d2_noisy_surface_retrieval(
            store,
            corpus_path=corpus,
        )

    if substage == "6e":
        from layers.layer6e_reconstruction import (
            run_layer6e_evidence_constrained_reconstruction,
        )

        return run_layer6e_evidence_constrained_reconstruction(store)

    if substage == "6f":
        from layers.layer6f_trust import (
            run_layer6f_semantic_transcription_trust,
        )

        return run_layer6f_semantic_transcription_trust(store)

    raise ValueError(f"Unsupported Stage 6 substage: {substage}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Stage-6 semantic substage inside its isolated "
            "native-ARM environment."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument(
        "--substage",
        required=True,
        choices=["6a", "6b", "6c", "6d", "6d2", "6e", "6f"],
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
    )
    parser.add_argument(
        "--vidyut-data",
        default="models/vidyut-0.4.0",
    )
    parser.add_argument(
        "--corpus",
        default="knowledge/stage6d_dcs/passages.jsonl",
    )

    args = parser.parse_args()

    artifacts = str(Path(args.artifacts).resolve())
    vidyut_data = str(Path(args.vidyut_data).resolve())
    corpus = str(Path(args.corpus).resolve())

    store = ArtifactStore(
        artifacts,
        args.run_id,
    )

    started = time.perf_counter()

    result = _dispatch(
        args.substage,
        store,
        vidyut_data=vidyut_data,
        corpus=corpus,
    )

    runtime_seconds = time.perf_counter() - started

    manifest = result.get("manifest", {}) if isinstance(result, dict) else {}
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}

    summary = {
        "runner_version": RUNNER_VERSION,
        "substage": args.substage,
        "run_id": args.run_id,
        "machine": platform.machine(),
        "python": sys.executable,
        "runtime_seconds": round(runtime_seconds, 3),
        "status": "ok",
        "layer_version": (
            manifest.get("version")
            or metrics.get("algorithm_version")
        ),
        "metrics": metrics,
        "next_action": manifest.get("next_action"),
    }

    summary_path = Path(store.run_dir) / "stage6_runtime" / (
        f"substage_{args.substage}_summary.json"
    )

    _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
