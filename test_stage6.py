from __future__ import annotations

import argparse
import json

from core.artifact_store import ArtifactStore
from layers.layer6_semantic import (
    LAYER6_VERSION,
    run_layer6_htr_evidence_parser,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Stage 6A test: ingest Stage-5 HTR evidence into the "
            "semantic/reconstruction pipeline without performing correction "
            "or translation."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument("--artifacts", default="artifacts")

    args = parser.parse_args()

    store = ArtifactStore(
        args.artifacts,
        args.run_id,
    )

    output = run_layer6_htr_evidence_parser(
        store,
    )

    print("Stage 6 version:", LAYER6_VERSION)
    print("Stage 6A HTR evidence parser completed.")
    print()

    print("Metrics:")
    for key, value in output.metrics.items():
        print(f"  {key}: {value}")

    print()
    print("First-line evidence:")

    first = output.lines[0]

    print(
        json.dumps(
            {
                "line_id": first.line_id,
                "reading_order": first.reading_order,
                "bbox": first.bbox,
                "H": first.htr_readiness_H,
                "agreement": first.cross_provider_agreement,
                "reconstruction_mode": first.reconstruction_mode,
                "evidence_status": first.evidence_status,
                "provider_a_readable": first.provider_a_top1_readable,
                "provider_b_readable": first.provider_b_top1_readable,
                "shared_readable_tokens": first.shared_readable_tokens,
                "provider_a_hypothesis_count": len(
                    first.provider_a_hypotheses
                ),
                "provider_b_hypothesis_count": len(
                    first.provider_b_hypotheses
                ),
                "notes": first.notes,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "Next action:",
        output.manifest.get("next_action"),
    )

    print()
    print("Artifacts:")
    print(
        f"  artifacts/{args.run_id}/L6/htr_input_table.json"
    )
    print(
        f"  artifacts/{args.run_id}/L6/htr_input_table.csv"
    )
    print(
        f"  artifacts/{args.run_id}/L6/page_sequence_candidates.json"
    )
    print(
        f"  artifacts/{args.run_id}/L6/stage6_manifest.json"
    )

    print()
    print(
        "IMPORTANT: Stage 6A prepares evidence only. "
        "It does not claim corrected Sanskrit, translation or T(p)."
    )


if __name__ == "__main__":
    main()
