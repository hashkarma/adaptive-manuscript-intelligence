from __future__ import annotations

import argparse
import json
import os

from core.artifact_store import ArtifactStore
from orchestration.evaluator import (
    evaluate_and_save_layer5,
    finalize_orchestration,
)


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required artifact is missing: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regression-test unified Stage 1-5 orchestration using existing "
            "artifacts. No HTR model inference is performed."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument("--artifacts", default="artifacts")

    args = parser.parse_args()

    run_dir = os.path.join(
        args.artifacts,
        args.run_id,
    )

    store = ArtifactStore(
        args.artifacts,
        args.run_id,
    )

    layer_reports = {}

    for layer_name in [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]:
        path = os.path.join(
            run_dir,
            "orchestration",
            f"{layer_name}_report.json",
        )
        layer_reports[layer_name] = _load_json(path)

    readiness_path = os.path.join(
        run_dir,
        "L5_readiness",
        "htr_readiness.json",
    )

    stage5_readiness = _load_json(readiness_path)

    layer5_report = evaluate_and_save_layer5(
        store,
        stage5_readiness,
        third_provider_available=False,
    )

    layer_reports["layer5"] = layer5_report

    final = finalize_orchestration(
        store,
        layer_reports,
    )

    print("Unified orchestration integration test completed.")
    print()
    print("Layer 5 report:")
    print(
        json.dumps(
            layer5_report,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Final orchestrator decision:")
    print(
        json.dumps(
            final,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Expected for the current validated run:")
    print("  phase = post_stage5")
    print("  S ≈ 0.9389")
    print("  H ≈ 0.3175")
    print("  htr_decision = scholar_review_required")
    print("  next_action = route_to_scholar_review")


if __name__ == "__main__":
    main()
