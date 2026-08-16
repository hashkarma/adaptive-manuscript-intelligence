from __future__ import annotations

import argparse
import json

from core.artifact_store import ArtifactStore
from layers.layer6c_morphology import (
    LAYER6C_VERSION,
    run_layer6c_morphology_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Stage 6C test: Vidyut morphology and "
            "derivational validation without promoting text."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument(
        "--artifacts",
        default="artifacts",
    )
    parser.add_argument(
        "--vidyut-data",
        default="models/vidyut-0.4.0",
    )

    args = parser.parse_args()

    store = ArtifactStore(
        args.artifacts,
        args.run_id,
    )

    output = run_layer6c_morphology_validation(
        store,
        vidyut_data_root=args.vidyut_data,
    )

    print("Stage 6C version:", LAYER6C_VERSION)
    print("Stage 6C morphology validation completed.")
    print()

    print("Metrics:")
    for key, value in output["metrics"].items():
        print(f"  {key}: {value}")

    first = output["lines"][0]

    print()
    print("First-line summary:")
    print(
        json.dumps(
            {
                "line_id": first["line_id"],
                "H": first["H"],
                "agreement": first[
                    "cross_provider_agreement"
                ],
                "line_conclusion": first[
                    "line_conclusion"
                ],
                "top_cluster_morphology": [
                    {
                        "cluster_id": row[
                            "cluster_id"
                        ],
                        "representative": row[
                            "representative_devanagari"
                        ],
                        "stage6b_score": row[
                            "stage6b_evidence_score"
                        ],
                        "stage6b_status": row[
                            "stage6b_evidence_status"
                        ],
                        "cross_provider_support": row[
                            "cross_provider_support"
                        ],
                        "stage6c_status": row[
                            "stage6c_status"
                        ],
                        "morphologically_analyzable": (
                            row["morphology"][
                                "morphologically_analyzable"
                            ]
                        ),
                        "exact_surface_rederived": (
                            row["morphology"][
                                "exact_surface_rederived"
                            ]
                        ),
                        "entry_kinds": row[
                            "morphology"
                        ]["entry_kinds"],
                        "lemmas": row[
                            "morphology"
                        ]["lemmas_devanagari"],
                    }
                    for row in first[
                        "validated_candidate_clusters"
                    ][:12]
                ],
                "sequence_morphology": [
                    {
                        "provider": row[
                            "provider"
                        ],
                        "rank": row[
                            "hypothesis_rank"
                        ],
                        "token_count": row[
                            "token_count"
                        ],
                        "morphology_coverage": row[
                            "morphology_coverage"
                        ],
                        "derivational_coverage": row[
                            "derivational_coverage"
                        ],
                    }
                    for row in first[
                        "sequence_morphology"
                    ]
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
        f"  artifacts/{args.run_id}/L6/morphology_evidence.json"
    )
    print(
        f"  artifacts/{args.run_id}/L6/stage6c_manifest.json"
    )

    print()
    print(
        "IMPORTANT: Morphological validity is linguistic evidence only. "
        "Stage 6C does not promote any candidate to final manuscript text."
    )


if __name__ == "__main__":
    main()
