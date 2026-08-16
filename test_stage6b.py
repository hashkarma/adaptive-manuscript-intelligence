from __future__ import annotations

import argparse
import json

from core.artifact_store import ArtifactStore
from layers.layer6b_reconstruct import (
    LAYER6B_VERSION,
    run_layer6b_candidate_reconstruction,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Stage 6B test: conservative Vidyut Viccheda, "
            "Kosha evidence, and observed-token candidate lattice."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument(
        "--vidyut-data",
        default="models/vidyut-0.4.0",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.72,
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.14,
    )

    args = parser.parse_args()

    store = ArtifactStore(
        args.artifacts,
        args.run_id,
    )

    output = run_layer6b_candidate_reconstruction(
        store,
        vidyut_data_root=args.vidyut_data,
        similarity_threshold=args.similarity_threshold,
        position_tolerance=args.position_tolerance,
    )

    print("Stage 6B version:", LAYER6B_VERSION)
    print("Stage 6B candidate lattice completed.")
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
                "reconstruction_mode": first["reconstruction_mode"],
                "observed_token_count": len(first["observed_tokens"]),
                "cheda_token_count": len(first["cheda_evidence"]),
                "candidate_cluster_count": len(first["candidate_clusters"]),
                "top_candidate_clusters": [
                    {
                        "cluster_id": cluster["cluster_id"],
                        "representative": cluster["representative_devanagari"],
                        "providers": cluster["provider_count"],
                        "hypotheses": cluster["hypothesis_count"],
                        "lexical_support_ratio": cluster["lexical_support_ratio"],
                        "cheda_support_ratio": cluster["cheda_support_ratio"],
                        "cross_provider_support": cluster["cross_provider_support"],
                        "mean_normalized_position": cluster.get(
                            "mean_normalized_position"
                        ),
                        "position_span": cluster.get("position_span"),
                        "line_evidence_gate": cluster.get(
                            "line_evidence_gate"
                        ),
                        "local_evidence_score": cluster.get(
                            "local_evidence_score"
                        ),
                        "evidence_score": cluster["evidence_score"],
                        "status": cluster["evidence_status"],
                        "members": [
                            {
                                "provider": member["provider"],
                                "rank": member["hypothesis_rank"],
                                "token": member["devanagari"],
                                "slp1": member["slp1"],
                                "kosha_exact": member["kosha_exact"],
                                "cheda_supported": member["cheda_supported"],
                            }
                            for member in cluster["members"]
                        ],
                    }
                    for cluster in first["candidate_clusters"][:12]
                ],
                "provisional_sequences": first[
                    "provisional_sequence_candidates"
                ][:6],
                "notes": first["notes"],
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
    print(f"  artifacts/{args.run_id}/L6/candidate_lattice.json")
    print(f"  artifacts/{args.run_id}/L6/viccheda_evidence.json")
    print(f"  artifacts/{args.run_id}/L6/stage6b_manifest.json")

    print()
    print(
        "IMPORTANT: Stage 6B ranks only observed HTR evidence plus "
        "Vidyut lexical/segmentation support. It does not emit final Sanskrit."
    )


if __name__ == "__main__":
    main()
