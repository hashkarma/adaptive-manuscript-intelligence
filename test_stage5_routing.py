from __future__ import annotations

import argparse
import json
import os

from orchestration.stage5_routing import (
    STAGE5_ROUTING_VERSION,
    evaluate_stage5_routing,
    load_readiness_artifact,
    write_stage5_routing_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5G standalone adaptive-routing validation. "
            "Consumes Stage 5F htr_readiness.json and writes an "
            "orchestrator decision artifact."
        )
    )

    parser.add_argument(
        "run_id"
    )

    parser.add_argument(
        "--artifacts",
        default="artifacts",
    )

    parser.add_argument(
        "--third-provider-available",
        action="store_true",
        help=(
            "Set only when a third HTR provider is actually implemented "
            "and runnable."
        ),
    )

    args = parser.parse_args()

    run_dir = os.path.join(
        args.artifacts,
        args.run_id,
    )

    readiness_path = os.path.join(
        run_dir,
        "L5_readiness",
        "htr_readiness.json",
    )

    output_path = os.path.join(
        run_dir,
        "orchestration",
        "stage5_decision.json",
    )

    print(
        "Stage 5G routing version:",
        STAGE5_ROUTING_VERSION,
    )

    print(
        "Readiness artifact:",
        readiness_path,
    )

    readiness = load_readiness_artifact(
        readiness_path
    )

    decision = evaluate_stage5_routing(
        readiness,
        third_provider_available=(
            args.third_provider_available
        ),
    )

    write_stage5_routing_artifact(
        output_path,
        decision,
    )

    print("\nStage 5 orchestration decision:")
    print(
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nDecision artifact:",
        output_path,
    )


if __name__ == "__main__":
    main()
