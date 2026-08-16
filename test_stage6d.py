from __future__ import annotations

import argparse
import json

from core.artifact_store import ArtifactStore
from layers.layer6d_context_rag import (
    LAYER6D_VERSION,
    run_layer6d_contextual_retrieval,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Stage 6D test: deterministic contextual retrieval "
            "against a local Sanskrit corpus."
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
        "--top-k-query",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--top-k-line",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    store = ArtifactStore(
        args.artifacts,
        args.run_id,
    )

    output = run_layer6d_contextual_retrieval(
        store,
        corpus_path=args.corpus,
        top_k_per_query=args.top_k_query,
        top_k_per_line=args.top_k_line,
    )

    print("Stage 6D version:", LAYER6D_VERSION)
    print("Stage 6D contextual retrieval completed.")
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
                "stage6c_conclusion": first[
                    "line_conclusion_from_stage6c"
                ],
                "best_context_evidence_score": first[
                    "best_context_evidence_score"
                ],
                "best_context_status": first[
                    "best_context_status"
                ],
                "top_context_hits": [
                    {
                        "source_file": hit[
                            "source_file"
                        ],
                        "passage_id": hit[
                            "passage_id"
                        ],
                        "surface_devanagari": hit[
                            "surface_devanagari"
                        ],
                        "retrieval_score": hit[
                            "retrieval_score"
                        ],
                        "context_evidence_score": hit[
                            "context_evidence_score"
                        ],
                        "status": hit[
                            "context_status"
                        ],
                        "independent_provider_query_support": hit[
                            "independent_provider_query_support"
                        ],
                    }
                    for hit in first[
                        "aggregated_context_hits"
                    ][:5]
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
        f"  artifacts/{args.run_id}/L6/context_retrieval.json"
    )
    print(
        f"  artifacts/{args.run_id}/L6/stage6d_manifest.json"
    )

    print()
    print(
        "IMPORTANT: retrieved corpus passages are contextual evidence only. "
        "Stage 6D does not emit final manuscript text."
    )


if __name__ == "__main__":
    main()
