from __future__ import annotations

import argparse
import json

from core.artifact_store import ArtifactStore
from layers.layer6e_reconstruction import (
    LAYER6E_VERSION,
    run_layer6e_evidence_constrained_reconstruction,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Stage 6E test: abstention-capable "
            "evidence-constrained Sanskrit reconstruction."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument(
        "--artifacts",
        default="artifacts",
    )

    args = parser.parse_args()

    store = ArtifactStore(
        args.artifacts,
        args.run_id,
    )

    output = (
        run_layer6e_evidence_constrained_reconstruction(
            store
        )
    )

    print(
        "Stage 6E version:",
        LAYER6E_VERSION,
    )
    print(
        "Stage 6E reconstruction completed."
    )
    print()

    print("Metrics:")
    for key, value in output["metrics"].items():
        print(f"  {key}: {value}")

    first = output["lines"][0]

    print()
    print("First-line result:")
    print(
        json.dumps(
            {
                "line_id": first["line_id"],
                "H": first["H"],
                "agreement": first[
                    "cross_provider_agreement"
                ],
                "best_stage6b_candidate_score": first[
                    "best_stage6b_candidate_score"
                ],
                "visually_and_morphologically_supported_candidates": first[
                    "visually_and_morphologically_supported_candidates"
                ],
                "best_context_score": first[
                    "best_context_score"
                ],
                "best_context_source": first[
                    "best_context_source"
                ],
                "reconstruction_evidence_index": first[
                    "reconstruction_evidence_index"
                ],
                "machine_decision": first[
                    "machine_decision"
                ],
                "diplomatic_transcription": first[
                    "diplomatic_transcription"
                ],
                "normalized_devanagari": first[
                    "normalized_devanagari"
                ],
                "supported_spans": first[
                    "supported_spans"
                ],
                "abstention_reasons": first[
                    "abstention_reasons"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "Next action:",
        output["manifest"].get(
            "next_action"
        ),
    )

    print()
    print("Artifacts:")
    print(
        f"  artifacts/{args.run_id}/L6/reconstruction_report.json"
    )
    print(
        f"  artifacts/{args.run_id}/L6/stage6e_manifest.json"
    )

    print()
    print(
        "IMPORTANT: Stage 6E may explicitly abstain. "
        "It does not generate unsupported Sanskrit and does not compute T(p)."
    )


if __name__ == "__main__":
    main()
