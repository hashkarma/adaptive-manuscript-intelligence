from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage5_runtime.coordinator import (
    Stage5PipelineError,
    run_stage5_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute Research Stage 5 end to end using isolated Provider-A "
            "and Provider-B Python environments."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--n-best", type=int, default=3)
    parser.add_argument("--max-output-length", type=int, default=192)
    parser.add_argument(
        "--provider-timeout-seconds",
        type=int,
        default=1800,
    )

    args = parser.parse_args()

    project_root = Path.cwd()

    try:
        summary = run_stage5_pipeline(
            project_root=project_root,
            run_id=args.run_id,
            artifacts=args.artifacts,
            device=args.device,
            num_beams=args.num_beams,
            n_best=args.n_best,
            max_output_length=args.max_output_length,
            provider_timeout_seconds=(
                args.provider_timeout_seconds
            ),
        )

    except Stage5PipelineError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "stage": exc.stage,
                    "error": str(exc),
                    "log_path": exc.log_path,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "status": "ok",
                **summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
