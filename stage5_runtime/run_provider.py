from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from core.artifact_store import ArtifactStore
import layers.layer5_htr as layer5


RUNTIME_PROVIDER_RUNNER_VERSION = "0.1.0-stage5-runtime-provider"


def _snapshot_l5(
    store: ArtifactStore,
    snapshot_name: str,
) -> str:
    source = Path(store.run_dir) / "L5"

    if not source.is_dir():
        raise FileNotFoundError(
            f"Stage 5 provider completed but L5 artifact directory "
            f"does not exist: {source}"
        )

    destination = Path(store.run_dir) / snapshot_name

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)

    return str(destination)


def _remove_working_l5(store: ArtifactStore) -> None:
    working = Path(store.run_dir) / "L5"

    if working.exists():
        shutil.rmtree(working)


def run_provider(
    *,
    run_id: str,
    artifacts: str,
    provider: str,
    snapshot_name: str,
    model_id: str | None,
    device: str,
    num_beams: int,
    n_best: int,
    max_output_length: int,
) -> Dict[str, Any]:
    store = ArtifactStore(artifacts, run_id)

    stage4_manifest = Path(store.run_dir) / "L4" / "line_manifest.json"

    if not stage4_manifest.exists():
        raise FileNotFoundError(
            f"Validated Stage 4 line manifest is missing: {stage4_manifest}"
        )

    _remove_working_l5(store)

    output = layer5.run_layer5_htr(
        store,
        provider_name=provider,
        model_id=model_id,
        device=device,
        num_beams=num_beams,
        n_best=n_best,
        max_output_length=max_output_length,
        max_lines=None,
    )

    metrics = dict(output.metrics)

    if int(metrics.get("attempted_lines", 0)) <= 0:
        raise RuntimeError(
            f"Provider {provider} attempted no Stage-4 lines."
        )

    if int(metrics.get("recognized_lines", 0)) <= 0:
        raise RuntimeError(
            f"Provider {provider} recognized no lines."
        )

    if int(metrics.get("error_lines", 0)) > 0:
        raise RuntimeError(
            f"Provider {provider} returned "
            f"{metrics.get('error_lines')} error line(s)."
        )

    snapshot_path = _snapshot_l5(
        store,
        snapshot_name,
    )

    manifest_path = (
        Path(snapshot_path)
        / "htr_manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Provider snapshot has no htr_manifest.json: {manifest_path}"
        )

    summary = {
        "runtime_runner_version": RUNTIME_PROVIDER_RUNNER_VERSION,
        "stage5_algorithm_version": layer5.LAYER5_VERSION,
        "run_id": run_id,
        "provider": metrics.get("provider"),
        "model_id": metrics.get("model_id"),
        "output_script": metrics.get("output_script"),
        "device": metrics.get("device"),
        "attempted_lines": metrics.get("attempted_lines"),
        "recognized_lines": metrics.get("recognized_lines"),
        "error_lines": metrics.get("error_lines"),
        "review_required_lines": metrics.get("review_required_lines"),
        "snapshot_name": snapshot_name,
        "snapshot_path": snapshot_path,
        "manifest_path": str(manifest_path),
        "htr_readiness_H": metrics.get("htr_readiness_H"),
        "cer": metrics.get("cer"),
        "wer": metrics.get("wer"),
        "python_executable": sys.executable,
    }

    with open(
        Path(snapshot_path) / "runtime_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one HTR provider in its isolated Python environment and "
            "persist a deterministic Stage-5 snapshot."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--n-best", type=int, default=3)
    parser.add_argument("--max-output-length", type=int, default=192)

    args = parser.parse_args()

    summary = run_provider(
        run_id=args.run_id,
        artifacts=args.artifacts,
        provider=args.provider,
        snapshot_name=args.snapshot,
        model_id=args.model,
        device=args.device,
        num_beams=args.num_beams,
        n_best=args.n_best,
        max_output_length=args.max_output_length,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
