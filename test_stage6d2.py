from __future__ import annotations

import argparse
import json

from core.artifact_store import ArtifactStore
from layers.layer6d2_noisy_retrieval import (
    LAYER6D2_VERSION,
    run_layer6d2_noisy_surface_retrieval,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Stage 6D.2 test: noisy Sanskrit HTR retrieval "
            "using RapidFuzz partial alignment."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument(
        "--artifacts",
        default="artifacts",
    )
    parser.add_argument(
        "--corpus",
        default="knowledge/stage6d_dcs/passages.jsonl",
    )
    parser.add_argument(
        "--preselect-k",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--top-k-query",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--top-k-line",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    store = ArtifactStore(
        args.artifacts,
        args.run_id,
    )

    output = run_layer6d2_noisy_surface_retrieval(
        store,
        corpus_path=args.corpus,
        preselect_k=args.preselect_k,
        top_k_query=args.top_k_query,
        top_k_line=args.top_k_line,
    )

    print("Stage 6D.2 version:", LAYER6D2_VERSION)
    print("Stage 6D.2 noisy-surface retrieval completed.")
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
                "agreement": first["cross_provider_agreement"],
                "stage6d1_best_context_score": first[
                    "stage6d1_best_context_score"
                ],
                "stage6d1_best_context_status": first[
                    "stage6d1_best_context_status"
                ],
                "best_noisy_surface_score": first[
                    "best_noisy_surface_score"
                ],
                "best_noisy_surface_status": first[
                    "best_noisy_surface_status"
                ],
                "top_hits": [
                    {
                        "source_file": hit["source_file"],
                        "passage_id": hit["passage_id"],
                        "surface_devanagari": hit[
                            "surface_devanagari"
                        ],
                        "partial_ratio": hit["partial_ratio"],
                        "wratio": hit["wratio"],
                        "full_ratio": hit["full_ratio"],
                        "fused_noisy_surface_score": hit[
                            "fused_noisy_surface_score"
                        ],
                        "status": hit["noisy_surface_status"],
                        "independent_provider_query_support": hit[
                            "independent_provider_query_support"
                        ],
                    }
                    for hit in first[
                        "aggregated_noisy_surface_hits"
                    ][:8]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "Next action:",
        output["manifest"].get("next_action"),
    )

    print()
    print("Artifacts:")
    print(
        f"  artifacts/{args.run_id}/L6/noisy_surface_retrieval.json"
    )
    print(
        f"  artifacts/{args.run_id}/L6/stage6d2_manifest.json"
    )

    print()
    print(
        "IMPORTANT: Stage 6D.2 retrieves attested noisy-surface neighbours "
        "only. It does not emit corrected Sanskrit."
    )


if __name__ == "__main__":
    main()
