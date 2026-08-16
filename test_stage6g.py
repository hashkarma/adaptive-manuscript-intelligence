from __future__ import annotations

import argparse
import json
import os

from core.artifact_store import ArtifactStore
from orchestration.evaluator import (
    evaluate_and_save_layer5,
    evaluate_and_save_layer6,
    finalize_orchestration,
)


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_optional(path: str) -> dict:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Stage 6G test. Refreshes Stage-5 routing under the "
            "new scholar-last policy, evaluates T(p), and produces the "
            "post-Stage-6 orchestration decision."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument(
        "--artifacts",
        default="artifacts",
    )

    parser.add_argument(
        "--third-provider-available",
        action="store_true",
    )
    parser.add_argument(
        "--alternate-visual-htr-available",
        action="store_true",
    )
    parser.add_argument(
        "--expanded-context-retry-available",
        action="store_true",
    )

    args = parser.parse_args()

    store = ArtifactStore(
        args.artifacts,
        args.run_id,
    )

    layer4_report = _load_json(
        store.path(
            "orchestration/layer4_report.json"
        )
    )

    stage5_readiness = _load_json(
        store.path(
            "L5_readiness/htr_readiness.json"
        )
    )

    stage6e = _load_json(
        store.path(
            "L6/stage6e_manifest.json"
        )
    )

    stage6f = _load_json(
        store.path(
            "L6/stage6f_manifest.json"
        )
    )

    retry_state = _load_optional(
        store.path(
            "orchestration/stage6_retry_state.json"
        )
    )

    # Refresh Stage-5 report so the old low-H -> scholar route is removed.
    layer5_report = evaluate_and_save_layer5(
        store,
        stage5_readiness,
        third_provider_available=(
            args.third_provider_available
        ),
    )

    layer6_report = evaluate_and_save_layer6(
        store,
        stage6f,
        stage6e,
        stage5_readiness,
        layer4_report,
        third_provider_available=(
            args.third_provider_available
        ),
        alternate_visual_htr_available=(
            args.alternate_visual_htr_available
        ),
        expanded_context_retry_available=(
            args.expanded_context_retry_available
        ),
        retry_state=retry_state,
    )

    reports = {
        "layer1": _load_json(
            store.path(
                "orchestration/layer1_report.json"
            )
        ),
        "layer2": _load_json(
            store.path(
                "orchestration/layer2_report.json"
            )
        ),
        "layer3": _load_json(
            store.path(
                "orchestration/layer3_report.json"
            )
        ),
        "layer4": layer4_report,
        "layer5": layer5_report,
        "layer6": layer6_report,
    }

    final = finalize_orchestration(
        store,
        reports,
    )

    print("Stage 6G adaptive routing completed.")
    print()

    print("Refreshed Stage-5 routing:")
    print(
        json.dumps(
            {
                "routing_version": layer5_report.get(
                    "routing_version"
                ),
                "S": layer5_report.get(
                    "signals",
                    {},
                ).get("S"),
                "H": layer5_report.get(
                    "signals",
                    {},
                ).get("H"),
                "decision": layer5_report.get(
                    "decision"
                ),
                "next_action": layer5_report.get(
                    "next_action"
                ),
                "scholar_review_required": layer5_report.get(
                    "scholar_review_required"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Stage-6G routing:")
    print(
        json.dumps(
            {
                "routing_version": layer6_report.get(
                    "routing_version"
                ),
                "signals": layer6_report.get(
                    "signals"
                ),
                "decision": layer6_report.get(
                    "decision"
                ),
                "next_action": layer6_report.get(
                    "next_action"
                ),
                "failure_domain": layer6_report.get(
                    "failure_domain"
                ),
                "selected_retry_action": layer6_report.get(
                    "selected_retry_action"
                ),
                "machine_retry_exhausted": layer6_report.get(
                    "machine_retry_exhausted"
                ),
                "machine_retry_exhaustion_reason": layer6_report.get(
                    "machine_retry_exhaustion_reason"
                ),
                "scholar_review_required": layer6_report.get(
                    "scholar_review_required"
                ),
                "retry_state": layer6_report.get(
                    "retry_state"
                ),
                "capabilities": layer6_report.get(
                    "capabilities"
                ),
                "recommendations": layer6_report.get(
                    "recommendations"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Unified final decision:")
    print(
        json.dumps(
            {
                "phase": final.get("phase"),
                "signals": final.get("signals"),
                "overall_status": final.get(
                    "overall_status"
                ),
                "semantic_decision": final.get(
                    "semantic_decision"
                ),
                "next_action": final.get(
                    "next_action"
                ),
                "failure_domain": final.get(
                    "failure_domain"
                ),
                "machine_retry_exhausted": final.get(
                    "machine_retry_exhausted"
                ),
                "scholar_review_required": final.get(
                    "scholar_review_required"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Artifacts:")
    print(
        f"  artifacts/{args.run_id}/orchestration/layer5_report.json"
    )
    print(
        f"  artifacts/{args.run_id}/orchestration/layer6_report.json"
    )
    print(
        f"  artifacts/{args.run_id}/orchestration/post_stage6_decision.json"
    )
    print(
        f"  artifacts/{args.run_id}/orchestration/final_decision.json"
    )


if __name__ == "__main__":
    main()
