from __future__ import annotations

import argparse
import json

from core.artifact_store import ArtifactStore
from layers.layer6f_trust import (
    LAYER6F_VERSION,
    run_layer6f_semantic_transcription_trust,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Stage 6F test: compute provisional semantic/"
            "transcription trust T(p)."
        )
    )
    parser.add_argument("run_id")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    store = ArtifactStore(args.artifacts, args.run_id)
    output = run_layer6f_semantic_transcription_trust(store)

    print("Stage 6F version:", LAYER6F_VERSION)
    print("Stage 6F semantic/transcription trust completed.")
    print()

    print("Metrics:")
    for key, value in output["metrics"].items():
        print(f"  {key}: {value}")

    first = output["lines"][0]

    print()
    print("First-line trust:")
    print(
        json.dumps(
            {
                "line_id": first["line_id"],
                "T": first["T"],
                "T_status": first["T_status"],
                "T_is_calibrated_probability": first[
                    "T_is_calibrated_probability"
                ],
                "components": first["components"],
                "weights": first["weights"],
                "machine_decision_from_stage6e": first[
                    "machine_decision_from_stage6e"
                ],
                "recommended_next_action": first[
                    "recommended_next_action"
                ],
                "abstention_reasons": first["abstention_reasons"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Next action:", output["manifest"].get("next_action"))
    print()
    print("Artifacts:")
    print(f"  artifacts/{args.run_id}/L6/transcription_trust.json")
    print(f"  artifacts/{args.run_id}/L6/stage6f_manifest.json")
    print()
    print(
        "IMPORTANT: T(p) is an uncalibrated system-trust/readiness signal. "
        "It is not accuracy, CER, WER, or probability of correctness."
    )


if __name__ == "__main__":
    main()
