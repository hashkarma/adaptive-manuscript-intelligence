from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage6_runtime.coordinator import (
    STAGE6_COORDINATOR_VERSION,
    Stage6PipelineError,
    run_stage6_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone test for the integrated Stage-6 isolated runtime."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument(
        "--project-root",
        default=".",
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
    )
    parser.add_argument(
        "--vidyut-data",
        default="models/vidyut-0.4.0",
    )
    parser.add_argument(
        "--corpus",
        default="knowledge/stage6d_dcs/passages.jsonl",
    )

    args = parser.parse_args()

    try:
        result = run_stage6_pipeline(
            project_root=Path(args.project_root),
            run_id=args.run_id,
            artifacts=args.artifacts,
            vidyut_data=args.vidyut_data,
            corpus=args.corpus,
        )
    except Stage6PipelineError as exc:
        print("Stage 6 runtime FAILED")
        print("stage:", exc.stage)
        print("detail:", str(exc))
        if exc.log_path:
            print("log:", exc.log_path)
        raise SystemExit(1)

    print(
        "Stage 6 coordinator version:",
        STAGE6_COORDINATOR_VERSION,
    )
    print("Stage 6 integrated runtime completed.")
    print()

    print(
        json.dumps(
            {
                "execution_mode": result.get(
                    "execution_mode"
                ),
                "runtime_seconds": result.get(
                    "runtime_seconds"
                ),
                "substage_plan": result.get(
                    "substage_plan"
                ),
                "trust": result.get(
                    "trust"
                ),
                "next_action": result.get(
                    "next_action"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Summary artifact:")
    print(
        f"  artifacts/{args.run_id}/stage6_runtime/"
        "stage6_pipeline_summary.json"
    )


if __name__ == "__main__":
    main()
