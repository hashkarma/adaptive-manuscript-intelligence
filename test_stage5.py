from __future__ import annotations

import argparse
import json
import os
import shutil

from core.artifact_store import ArtifactStore
import layers.layer5_htr as layer5


def _find_base_dir(run_id: str, requested: str | None) -> str:
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend(["artifacts", "."])

    checked = []
    for base_dir in candidates:
        candidate = os.path.join(
            base_dir,
            run_id,
            "L4",
            "line_manifest.json",
        )
        checked.append(candidate)
        if os.path.exists(candidate):
            return base_dir

    raise FileNotFoundError(
        "Could not locate a validated Stage 4 run. Checked:\n  "
        + "\n  ".join(checked)
    )


def _fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Stage 5 provider through the common quality contract."
    )

    parser.add_argument("run_id")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument(
        "--provider",
        default=layer5.DEFAULT_PROVIDER,
        choices=[
            "trocr_iast_baseline",
            "trocr_baseline",
            "trocr_vedic_devanagari",
        ],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--n-best", type=int, default=3)
    parser.add_argument("--max-output-length", type=int, default=192)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--no-backup", action="store_true")

    args = parser.parse_args()

    print("Loaded module:", os.path.abspath(layer5.__file__))
    print("Stage 5 version:", layer5.LAYER5_VERSION)
    print("Provider:", args.provider)

    base_dir = _find_base_dir(args.run_id, args.artifacts)
    print("Artifact base directory:", os.path.abspath(base_dir))

    store = ArtifactStore(base_dir, args.run_id)

    l5_dir = os.path.join(store.run_dir, "L5")
    if os.path.isdir(l5_dir):
        if not args.no_backup:
            backup_dir = l5_dir + "_previous"
            if os.path.isdir(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(l5_dir, backup_dir)
            print("Previous L5 backed up to:", backup_dir)
        shutil.rmtree(l5_dir)

    print()
    print("IMPORTANT:")
    print("This is a provider comparison/smoke-test step.")
    print("It does not establish scholarly correctness.")
    print("CER/WER and final H(p) remain unavailable.")
    print()

    output = layer5.run_layer5_htr(
        store,
        provider_name=args.provider,
        model_id=args.model,
        device=args.device,
        num_beams=args.num_beams,
        n_best=args.n_best,
        max_output_length=args.max_output_length,
        max_lines=args.max_lines,
    )

    print("\nStage 5 provider run completed")

    print("\nMetrics:")
    for name, value in output.metrics.items():
        print(f"  {name}: {_fmt(value)}")

    print("\nLine results:")
    for line in sorted(output.lines, key=lambda row: row.reading_order):
        print()
        print(
            f"[{line.line_id}] "
            f"status={line.status} "
            f"review={line.review_required} "
            f"device={line.device_used}"
        )

        if line.error:
            print("  ERROR:", line.error)
            continue

        print("  raw_script:", line.raw_script)
        print("  raw_text:", line.raw_text)
        print("  devanagari:", line.devanagari_text)

        q = line.quality
        print("  quality:")
        print("    hypothesis_entropy:", _fmt(q.hypothesis_entropy))
        print("    decoder_stability:", _fmt(q.decoder_stability))
        print("    script_purity:", _fmt(q.script_purity))
        print("    token_repetition_ratio:", _fmt(q.token_repetition_ratio))
        print(
            "    hypothesis_length_spread:",
            _fmt(q.hypothesis_length_spread),
        )

        if q.warnings:
            print("    warnings:")
            for warning in q.warnings:
                print("      -", warning)
        else:
            print("    warnings: none")

        if len(line.hypotheses) > 1:
            print("  alternatives:")
            for hyp in line.hypotheses[1:]:
                print(
                    f"    {hyp.rank}. {hyp.raw_text} "
                    f"(relative_score={hyp.relative_score})"
                )

    summary = {
        "provider": output.metrics["provider"],
        "model_id": output.metrics["model_id"],
        "output_script": output.metrics["output_script"],
        "attempted_lines": output.metrics["attempted_lines"],
        "recognized_lines": output.metrics["recognized_lines"],
        "error_lines": output.metrics["error_lines"],
        "review_required_lines": output.metrics["review_required_lines"],
        "mean_hypothesis_entropy": output.metrics["mean_hypothesis_entropy"],
        "mean_decoder_stability": output.metrics["mean_decoder_stability"],
        "mean_script_purity": output.metrics["mean_script_purity"],
        "htr_readiness_H": output.metrics["htr_readiness_H"],
        "cer": output.metrics["cer"],
        "wer": output.metrics["wer"],
    }

    print("\nCompact provider summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
