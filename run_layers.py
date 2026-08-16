from __future__ import annotations
import argparse
import cv2

from core.artifact_store import ArtifactStore
from layers.layer0_ingest import run_layer0_ingest
from layers.layer1_restore import run_layer1_restore
from layers.layer2_damage import run_layer2_damage
from layers.layer3_layout import run_layer3_layout

from orchestration.evaluator import (
    evaluate_and_save_layer1,
    evaluate_and_save_layer2,
    evaluate_and_save_layer3,
    finalize_orchestration,
)


def main():
    parser = argparse.ArgumentParser(description="Run manuscript pipeline with orchestration.")
    parser.add_argument("--input", required=True, help="Path to manuscript image")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--notes", default="manuscript demo run")
    args = parser.parse_args()

    run_id = ArtifactStore.new_run_id("manuscript")
    store = ArtifactStore(args.artifacts, run_id)

    print(f"\n=== RUN ID: {run_id} ===")
    print(f"Artifacts folder: {store.run_dir}")

    # L0
    print("\n[L0] Ingestion...")
    l0 = run_layer0_ingest(args.input, store, notes=args.notes)
    print("  ✓ raw image stored")

    # L1
    print("\n[L1] Visual restoration...")
    l1 = run_layer1_restore(l0.raw_bgr, store)
    report1 = evaluate_and_save_layer1(store, l1.metrics)
    print(f"  ✓ status: {report1['status']}")
    if report1["complaints"]:
        print("  complaints:", "; ".join(report1["complaints"]))

    balanced_gray = cv2.cvtColor(l1.balanced_bgr, cv2.COLOR_BGR2GRAY)

    # L2
    print("\n[L2] Damage & uncertainty...")
    l2 = run_layer2_damage(balanced_gray, store)
    report2 = evaluate_and_save_layer2(store, l2.metrics)
    print(f"  ✓ status: {report2['status']}")
    if report2["complaints"]:
        print("  complaints:", "; ".join(report2["complaints"]))

    # L3
    print("\n[L3] Layout detection...")
    l3 = run_layer3_layout(balanced_gray, l1.binary_u8, store)
    report3 = evaluate_and_save_layer3(store, l3.metrics)
    print(f"  ✓ status: {report3['status']}")
    if report3["complaints"]:
        print("  complaints:", "; ".join(report3["complaints"]))

    # orchestration summary
    print("\n[ORCHESTRATION] Final evaluation...")
    final = finalize_orchestration(
        store,
        {
            "layer1": report1,
            "layer2": report2,
            "layer3": report3,
        },
    )

    print(f"  overall_status: {final['overall_status']}")
    print(f"  next_action   : {final['next_action']}")
    print(f"  summary       : {final['summary']}")
    if final["complaints"]:
        print("  complaints    :")
        for c in final["complaints"]:
            print("   -", c)

    print("\nSaved orchestration outputs:")
    print(f"  {store.run_dir}/orchestration/layer1_report.json")
    print(f"  {store.run_dir}/orchestration/layer2_report.json")
    print(f"  {store.run_dir}/orchestration/layer3_report.json")
    print(f"  {store.run_dir}/orchestration/final_decision.json")


if __name__ == "__main__":
    main()