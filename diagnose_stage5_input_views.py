from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


DIAGNOSTIC_VERSION = "0.1.0-stage5-input-view-ablation"

PROVIDERS = {
    "provider_a": {
        "provider": "trocr_iast_baseline",
        "python_rel": "venv-stage5/bin/python",
        "snapshot": "L5_provider_A",
    },
    "provider_b": {
        "provider": "trocr_vedic_devanagari",
        "python_rel": "venv-provider-b/bin/python",
        "snapshot": "L5_provider_B",
    },
}

VIEWS = (
    "raw",
    "balanced",
    "binary",
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Image is missing or unreadable: {path}")

    return image


def _image_hw(image: np.ndarray) -> Tuple[int, int]:
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return int(image.shape[0]), int(image.shape[1])


def _crop_page_geometry(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> np.ndarray:
    image_h, image_w = _image_hw(image)

    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(image_w, x0 + int(w))
    y1 = min(image_h, y0 + int(h))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Invalid crop after bounds checking: "
            f"x={x}, y={y}, w={w}, h={h}, image={image.shape}"
        )

    return image[y0:y1, x0:x1].copy()


def _background_value(image: np.ndarray) -> int | Tuple[int, int, int]:
    """
    Estimate a neutral light border for synthetic HTR context.

    This adds context without importing pixels from neighbouring physical lines.
    """
    if image.ndim == 2:
        value = int(
            np.clip(
                np.percentile(image, 95),
                220,
                255,
            )
        )
        return value

    if image.ndim == 3:
        values = []

        for channel in range(image.shape[2]):
            value = int(
                np.clip(
                    np.percentile(image[:, :, channel], 95),
                    220,
                    255,
                )
            )
            values.append(value)

        # cv2 uses BGR order; copyMakeBorder accepts the same scalar order.
        return tuple(values[:3])

    return 255


def _htr_safe_crop(
    crop: np.ndarray,
    *,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    return cv2.copyMakeBorder(
        crop,
        int(pad_y),
        int(pad_y),
        int(pad_x),
        int(pad_x),
        borderType=cv2.BORDER_CONSTANT,
        value=_background_value(crop),
    )


def _provider_command_prefix(python_path: Path) -> List[str]:
    """
    Force native ARM64 provider execution on Apple Silicon.

    The main Stage 1-4 environment may be x86_64/Rosetta; the HTR virtual
    environments contain native ARM64 compiled wheels.
    """
    if (
        sys.platform == "darwin"
        and Path("/usr/bin/arch").exists()
    ):
        return [
            "/usr/bin/arch",
            "-arm64",
            str(python_path),
        ]

    return [str(python_path)]


def _run_provider(
    *,
    project_root: Path,
    artifacts_base: Path,
    workspace_run_id: str,
    provider_key: str,
    device: str,
    timeout_seconds: int,
    log_path: Path,
) -> None:
    config = PROVIDERS[provider_key]

    python_path = (
        project_root
        / config["python_rel"]
    )

    if not python_path.exists():
        raise FileNotFoundError(
            f"{provider_key} Python is missing: {python_path}"
        )

    command = [
        *_provider_command_prefix(python_path),
        "-m",
        "stage5_runtime.run_provider",
        workspace_run_id,
        "--artifacts",
        str(artifacts_base),
        "--provider",
        config["provider"],
        "--snapshot",
        config["snapshot"],
        "--device",
        device,
        "--num-beams",
        "4",
        "--n-best",
        "3",
        "--max-output-length",
        "192",
    ]

    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )

    except subprocess.TimeoutExpired as exc:
        output = (
            exc.stdout
            if isinstance(exc.stdout, str)
            else ""
        )

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")

        raise RuntimeError(
            f"{provider_key} timed out after {timeout_seconds}s. "
            f"Log: {log_path}"
        ) from exc

    elapsed = time.perf_counter() - started

    log_text = (
        f"$ {' '.join(command)}\n\n"
        f"{completed.stdout or ''}\n\n"
        f"[exit_code={completed.returncode}] "
        f"[runtime_seconds={elapsed:.3f}]\n"
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")

    if completed.returncode != 0:
        tail = "\n".join(
            (completed.stdout or "").splitlines()[-30:]
        )

        raise RuntimeError(
            f"{provider_key} failed with exit code "
            f"{completed.returncode}.\n{tail}\nLog: {log_path}"
        )


def _find_line(
    manifest: Dict[str, Any],
    line_id: str,
) -> Dict[str, Any]:
    for line in manifest.get("lines", []):
        if str(line.get("line_id")) == line_id:
            return dict(line)

    available = [
        str(line.get("line_id"))
        for line in manifest.get("lines", [])
    ]

    raise KeyError(
        f"Line {line_id!r} was not found in the Stage-4 manifest. "
        f"Available: {available}"
    )


def _prepare_workspace(
    *,
    source_run_dir: Path,
    ablation_root: Path,
    line_id: str,
) -> Dict[str, Any]:
    source_manifest_path = (
        source_run_dir
        / "L4"
        / "line_manifest.json"
    )

    manifest = _load_json(source_manifest_path)
    line = _find_line(manifest, line_id)

    x = int(line["x"])
    y = int(line["y"])
    width = int(line["w"])
    height = int(line["h"])

    pad_x = int(
        line.get(
            "htr_padding_x",
            12,
        )
    )

    pad_y = int(
        line.get(
            "htr_padding_y",
            max(
                4,
                round(height * 0.12),
            ),
        )
    )

    source_paths = {
        "raw": source_run_dir / "L0" / "raw.png",
        "balanced": source_run_dir / "L1" / "balanced.png",
        "binary": source_run_dir / "L1" / "binary.png",
    }

    source_images = {
        view: _load_image(path)
        for view, path in source_paths.items()
    }

    shapes = {
        view: _image_hw(image)
        for view, image in source_images.items()
    }

    if len(set(shapes.values())) != 1:
        raise ValueError(
            "RAW/BALANCED/BINARY page dimensions differ; "
            f"same-geometry ablation would be invalid: {shapes}"
        )

    workspace_base = (
        ablation_root
        / "workspace"
    )

    workspace_run_id = (
        f"{line_id}_three_views"
    )

    workspace_run_dir = (
        workspace_base
        / workspace_run_id
    )

    if workspace_run_dir.exists():
        shutil.rmtree(workspace_run_dir)

    crop_dir = (
        workspace_run_dir
        / "L4"
        / "lines"
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_dir = (
        ablation_root
        / "inputs"
    )

    if input_dir.exists():
        shutil.rmtree(input_dir)

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_lines = []
    inputs = {}

    for index, view in enumerate(VIEWS, start=1):
        core = _crop_page_geometry(
            source_images[view],
            x=x,
            y=y,
            w=width,
            h=height,
        )

        htr_crop = _htr_safe_crop(
            core,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        diagnostic_line_id = (
            f"{line_id}_{view}"
        )

        rel_path = (
            f"L4/lines/{diagnostic_line_id}.png"
        )

        workspace_crop_path = (
            workspace_run_dir
            / rel_path
        )

        ok = cv2.imwrite(
            str(workspace_crop_path),
            htr_crop,
        )

        if not ok:
            raise RuntimeError(
                f"Failed to save diagnostic crop: {workspace_crop_path}"
            )

        # Also save a user-friendly copy in the source run's diagnostic area.
        friendly_path = (
            input_dir
            / f"{view}_{line_id}.png"
        )

        ok = cv2.imwrite(
            str(friendly_path),
            htr_crop,
        )

        if not ok:
            raise RuntimeError(
                f"Failed to save diagnostic input: {friendly_path}"
            )

        diagnostic_line = copy.deepcopy(line)
        diagnostic_line["line_id"] = diagnostic_line_id
        diagnostic_line["reading_order"] = index
        diagnostic_line["crop_rel_path"] = rel_path
        diagnostic_line["ablation_view"] = view
        diagnostic_line["source_line_id"] = line_id
        diagnostic_line["htr_crop_width"] = int(htr_crop.shape[1])
        diagnostic_line["htr_crop_height"] = int(htr_crop.shape[0])
        diagnostic_line["htr_padding_x"] = pad_x
        diagnostic_line["htr_padding_y"] = pad_y

        diagnostic_lines.append(
            diagnostic_line
        )

        inputs[view] = {
            "source_image": str(source_paths[view]),
            "diagnostic_crop": str(friendly_path),
            "workspace_crop": str(workspace_crop_path),
            "page_geometry": {
                "x": x,
                "y": y,
                "w": width,
                "h": height,
            },
            "padding": {
                "x": pad_x,
                "y": pad_y,
            },
            "crop_shape": list(htr_crop.shape),
        }

    workspace_manifest = copy.deepcopy(manifest)
    workspace_manifest["stage"] = "stage4_ablation_input"
    workspace_manifest["run_id"] = workspace_run_id
    workspace_manifest["lines"] = diagnostic_lines

    metrics = dict(
        workspace_manifest.get(
            "metrics",
            {},
        )
    )

    metrics["num_lines"] = len(
        diagnostic_lines
    )
    metrics["ablation_source_run"] = (
        source_run_dir.name
    )
    metrics["ablation_source_line"] = line_id
    metrics["ablation_views"] = list(VIEWS)

    workspace_manifest["metrics"] = metrics

    workspace_manifest["ablation"] = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "purpose": (
            "Hold physical line geometry constant while varying only "
            "the visual representation supplied to Stage-5 HTR."
        ),
        "source_run": source_run_dir.name,
        "source_line": line_id,
        "views": list(VIEWS),
        "same_page_geometry_for_all_views": True,
    }

    _write_json(
        workspace_run_dir
        / "L4"
        / "line_manifest.json",
        workspace_manifest,
    )

    return {
        "workspace_base": workspace_base,
        "workspace_run_id": workspace_run_id,
        "workspace_run_dir": workspace_run_dir,
        "inputs": inputs,
        "source_line": line,
        "source_manifest_version": manifest.get("version"),
    }


def _normalise_for_reference(text: str) -> str:
    """
    Normalize for diagnostic similarity only.

    Exact HTR text remains untouched in artifacts/results.
    """
    text = unicodedata.normalize(
        "NFC",
        text or "",
    )

    kept: List[str] = []

    for ch in text:
        cp = ord(ch)

        # Ignore Vedic / extension marks for this modern-readable-prefix
        # diagnostic. Preserve Devanagari letters and dependent signs.
        if (
            0x1CD0 <= cp <= 0x1CFF
            or 0xA8E0 <= cp <= 0xA8FF
            or 0x0300 <= cp <= 0x036F
        ):
            continue

        category = unicodedata.category(ch)

        if ch.isspace():
            continue

        if category.startswith("P"):
            continue

        if category.startswith("C"):
            continue

        kept.append(ch)

    return "".join(kept)


def _best_window_similarity(
    output_text: str,
    reference_text: str,
) -> float | None:
    output = _normalise_for_reference(
        output_text
    )

    reference = _normalise_for_reference(
        reference_text
    )

    if not reference:
        return None

    if not output:
        return 0.0

    if reference in output:
        return 1.0

    ref_len = len(reference)

    min_window = max(
        1,
        int(round(ref_len * 0.75)),
    )

    max_window = min(
        len(output),
        max(
            min_window,
            int(round(ref_len * 1.35)),
        ),
    )

    if len(output) <= max_window:
        return round(
            difflib.SequenceMatcher(
                None,
                output,
                reference,
            ).ratio(),
            4,
        )

    best = 0.0

    for window_len in range(
        min_window,
        max_window + 1,
    ):
        for start in range(
            0,
            len(output)
            - window_len
            + 1,
        ):
            window = output[
                start:
                start + window_len
            ]

            score = (
                difflib.SequenceMatcher(
                    None,
                    window,
                    reference,
                ).ratio()
            )

            if score > best:
                best = score

    return round(best, 4)


def _extract_line_results(
    manifest: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for line in manifest.get("lines", []):
        line_id = str(
            line.get("line_id", "")
        )

        if not line_id:
            continue

        quality = (
            line.get("quality")
            or {}
        )

        result[line_id] = {
            "status": line.get("status"),
            "review_required": line.get(
                "review_required",
                line.get("review", False),
            ),
            "raw_script": line.get(
                "raw_script"
            ),
            "raw_text": line.get(
                "raw_text",
                line.get("raw_iast", ""),
            ),
            "transliteration_input": line.get(
                "transliteration_input"
            ),
            "devanagari_text": line.get(
                "devanagari_text",
                line.get("devanagari", ""),
            ),
            "hypothesis_entropy": quality.get(
                "hypothesis_entropy"
            ),
            "decoder_stability": quality.get(
                "decoder_stability"
            ),
            "script_purity": quality.get(
                "script_purity"
            ),
            "token_repetition_ratio": quality.get(
                "token_repetition_ratio"
            ),
            "hypothesis_length_spread": quality.get(
                "hypothesis_length_spread"
            ),
            "warnings": quality.get(
                "warnings",
                [],
            ),
            "alternatives": line.get(
                "alternatives",
                line.get(
                    "hypotheses",
                    [],
                ),
            ),
        }

    return result


def _provider_manifest_path(
    workspace_run_dir: Path,
    provider_key: str,
) -> Path:
    return (
        workspace_run_dir
        / PROVIDERS[provider_key]["snapshot"]
        / "htr_manifest.json"
    )


def _print_summary(
    results: Dict[str, Any],
) -> None:
    print()
    print("=" * 100)
    print("STAGE-5 INPUT-VIEW ABLATION")
    print("=" * 100)
    print(
        "Same physical line geometry is used for RAW, BALANCED and BINARY."
    )
    print(
        "Only the visual representation changes; Provider A/B code is unchanged."
    )

    reference = results.get(
        "reference_prefix"
    )

    if reference:
        print()
        print(
            "Diagnostic reference prefix:"
        )
        print(reference)

    for view in VIEWS:
        print()
        print("-" * 100)
        print(f"VIEW: {view.upper()}")

        for provider_key in (
            "provider_a",
            "provider_b",
        ):
            item = (
                results["results"]
                [view]
                [provider_key]
            )

            print()
            print(
                f"{provider_key.upper()} "
                f"({item['provider']})"
            )
            print(
                "Devanagari:"
            )
            print(
                item.get(
                    "devanagari_text"
                )
                or "<blank>"
            )

            print(
                "decoder_stability:",
                item.get(
                    "decoder_stability"
                ),
            )
            print(
                "script_purity:",
                item.get(
                    "script_purity"
                ),
            )

            if reference:
                print(
                    "reference_best_window_similarity:",
                    item.get(
                        "reference_best_window_similarity"
                    ),
                )

    print()
    print("=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print(
        "RAW clearly better -> investigate Stage 1 enhancement."
    )
    print(
        "BALANCED clearly better -> Stage 1 balanced view is helping HTR."
    )
    print(
        "BINARY clearly better -> consider a binary HTR input path."
    )
    print(
        "All three badly wrong -> dominant problem is Stage-5 model/domain fit, not Stage 1/4."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test RAW vs BALANCED vs BINARY visual representations "
            "of the same Stage-4 physical line through Provider A and B."
        )
    )

    parser.add_argument(
        "run_id",
        help="Existing manuscript run ID.",
    )

    parser.add_argument(
        "--artifacts",
        default="artifacts",
    )

    parser.add_argument(
        "--line-id",
        default="line_001",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "mps",
            "cuda",
            "cpu",
        ],
    )

    parser.add_argument(
        "--reference-prefix",
        default="",
        help=(
            "Optional known visible prefix used only for diagnostic similarity; "
            "this does not become training ground truth automatically."
        ),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1200,
    )

    args = parser.parse_args()

    project_root = Path.cwd().resolve()

    artifacts_path = Path(
        args.artifacts
    )

    if not artifacts_path.is_absolute():
        artifacts_path = (
            project_root
            / artifacts_path
        )

    artifacts_path = artifacts_path.resolve()

    source_run_dir = (
        artifacts_path
        / args.run_id
    )

    if not source_run_dir.is_dir():
        raise FileNotFoundError(
            f"Run does not exist: {source_run_dir}"
        )

    ablation_root = (
        source_run_dir
        / "stage5_input_ablation"
    )

    ablation_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared = _prepare_workspace(
        source_run_dir=source_run_dir,
        ablation_root=ablation_root,
        line_id=args.line_id,
    )

    workspace_base = Path(
        prepared["workspace_base"]
    )

    workspace_run_id = str(
        prepared["workspace_run_id"]
    )

    workspace_run_dir = Path(
        prepared["workspace_run_dir"]
    )

    logs_dir = (
        ablation_root
        / "logs"
    )

    started = time.time()

    for provider_key in (
        "provider_a",
        "provider_b",
    ):
        _run_provider(
            project_root=project_root,
            artifacts_base=workspace_base,
            workspace_run_id=workspace_run_id,
            provider_key=provider_key,
            device=args.device,
            timeout_seconds=args.timeout_seconds,
            log_path=(
                logs_dir
                / f"{provider_key}.log"
            ),
        )

    provider_manifests = {
        provider_key: _load_json(
            _provider_manifest_path(
                workspace_run_dir,
                provider_key,
            )
        )
        for provider_key in (
            "provider_a",
            "provider_b",
        )
    }

    provider_lines = {
        provider_key: _extract_line_results(
            provider_manifests[
                provider_key
            ]
        )
        for provider_key in (
            "provider_a",
            "provider_b",
        )
    }

    results_by_view: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for view in VIEWS:
        diagnostic_line_id = (
            f"{args.line_id}_{view}"
        )

        results_by_view[view] = {}

        for provider_key in (
            "provider_a",
            "provider_b",
        ):
            if (
                diagnostic_line_id
                not in provider_lines[
                    provider_key
                ]
            ):
                raise KeyError(
                    f"{diagnostic_line_id} missing from "
                    f"{provider_key} manifest."
                )

            item = dict(
                provider_lines[
                    provider_key
                ][
                    diagnostic_line_id
                ]
            )

            item["provider"] = (
                PROVIDERS[
                    provider_key
                ]["provider"]
            )

            item["view"] = view

            item[
                "reference_best_window_similarity"
            ] = (
                _best_window_similarity(
                    item.get(
                        "devanagari_text",
                        "",
                    ),
                    args.reference_prefix,
                )
                if args.reference_prefix
                else None
            )

            results_by_view[
                view
            ][
                provider_key
            ] = item

    output = {
        "diagnostic_version": (
            DIAGNOSTIC_VERSION
        ),
        "source_run_id": args.run_id,
        "source_stage4_version": (
            prepared[
                "source_manifest_version"
            ]
        ),
        "source_line_id": args.line_id,
        "purpose": (
            "Determine whether Stage-1 visual representation or Stage-5 "
            "model/domain fit is the dominant cause of recognition failure."
        ),
        "controlled_variables": {
            "same_physical_line_geometry": True,
            "same_htr_context_padding": True,
            "same_provider_decoding_parameters": True,
            "variable_under_test": (
                "raw_vs_balanced_vs_binary_visual_representation"
            ),
        },
        "reference_prefix": (
            args.reference_prefix
            or None
        ),
        "reference_note": (
            "Used only as diagnostic comparison evidence; not automatically "
            "treated as scholar-certified ground truth."
            if args.reference_prefix
            else None
        ),
        "inputs": prepared["inputs"],
        "providers": {
            key: {
                "provider": value["provider"],
                "python_environment": value["python_rel"],
            }
            for key, value in PROVIDERS.items()
        },
        "results": results_by_view,
        "runtime_seconds": round(
            time.time() - started,
            3,
        ),
        "artifacts": {
            "inputs_dir": str(
                ablation_root / "inputs"
            ),
            "provider_a_log": str(
                logs_dir / "provider_a.log"
            ),
            "provider_b_log": str(
                logs_dir / "provider_b.log"
            ),
            "workspace": str(
                workspace_run_dir
            ),
        },
    }

    output_path = (
        ablation_root
        / "ablation_results.json"
    )

    _write_json(
        output_path,
        output,
    )

    _print_summary(
        output
    )

    print()
    print(
        "Result artifact:",
        output_path,
    )


if __name__ == "__main__":
    main()
