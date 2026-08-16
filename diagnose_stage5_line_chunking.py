from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


DIAGNOSTIC_VERSION = "0.1.0-stage5-long-line-chunking"

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

VIEWS = ("raw", "balanced")
VARIANTS = ("full", "2chunk", "3chunk")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image is missing or unreadable: {path}")
    return image


def _image_hw(image: np.ndarray) -> Tuple[int, int]:
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return int(image.shape[0]), int(image.shape[1])


def _crop(
    image: np.ndarray,
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
            f"Invalid crop: x={x}, y={y}, w={w}, h={h}, image={image.shape}"
        )

    return image[y0:y1, x0:x1].copy()


def _background_value(image: np.ndarray) -> int | Tuple[int, int, int]:
    if image.ndim == 2:
        return int(np.clip(np.percentile(image, 95), 220, 255))

    values = []
    for channel in range(min(3, image.shape[2])):
        values.append(
            int(
                np.clip(
                    np.percentile(image[:, :, channel], 95),
                    220,
                    255,
                )
            )
        )
    return tuple(values)


def _pad_for_htr(
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


def _ink_mask(binary_crop: np.ndarray) -> np.ndarray:
    if binary_crop.ndim == 3:
        binary_crop = cv2.cvtColor(binary_crop, cv2.COLOR_BGR2GRAY)

    dark = binary_crop < 128
    light = binary_crop >= 128

    # Text is normally the minority class in a manuscript line crop.
    return dark if np.count_nonzero(dark) <= np.count_nonzero(light) else light


def _smooth_1d(values: np.ndarray, window: int = 7) -> np.ndarray:
    window = max(3, int(window) | 1)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _choose_split(
    col_profile: np.ndarray,
    ideal_x: int,
    *,
    search_radius: int,
    forbidden_margin: int,
    existing_splits: Sequence[int],
    min_split_gap: int,
) -> int:
    width = len(col_profile)

    lo = max(
        forbidden_margin,
        int(ideal_x) - int(search_radius),
    )
    hi = min(
        width - forbidden_margin - 1,
        int(ideal_x) + int(search_radius),
    )

    if hi <= lo:
        return int(np.clip(ideal_x, forbidden_margin, width - forbidden_margin - 1))

    candidates: List[Tuple[float, float, int]] = []

    denom = max(1.0, float(search_radius))

    for x in range(lo, hi + 1):
        if any(abs(x - s) < min_split_gap for s in existing_splits):
            continue

        ink_score = float(col_profile[x])

        # Prefer low-ink columns, but gently bias toward the intended fraction.
        distance_penalty = abs(x - ideal_x) / denom
        score = ink_score + 0.20 * distance_penalty

        candidates.append((score, ink_score, x))

    if not candidates:
        return int(np.clip(ideal_x, forbidden_margin, width - forbidden_margin - 1))

    candidates.sort(key=lambda item: (item[0], item[1], abs(item[2] - ideal_x)))
    return int(candidates[0][2])


def _discover_splits(
    binary_core: np.ndarray,
    chunk_count: int,
) -> List[int]:
    if chunk_count <= 1:
        return []

    ink = _ink_mask(binary_core)
    col_profile = np.sum(ink, axis=0).astype(np.float32)
    col_profile = _smooth_1d(col_profile, 7)

    width = int(binary_core.shape[1])

    forbidden_margin = max(24, int(round(width * 0.08)))
    search_radius = max(30, int(round(width * 0.12)))
    min_split_gap = max(80, int(round(width * 0.18)))

    splits: List[int] = []

    for index in range(1, chunk_count):
        ideal = int(round(width * index / chunk_count))

        split = _choose_split(
            col_profile,
            ideal,
            search_radius=search_radius,
            forbidden_margin=forbidden_margin,
            existing_splits=splits,
            min_split_gap=min_split_gap,
        )

        splits.append(split)

    splits = sorted(set(splits))

    if len(splits) != chunk_count - 1:
        raise RuntimeError(
            f"Could not derive {chunk_count - 1} distinct split points; got {splits}"
        )

    return splits


def _chunk_ranges(
    width: int,
    splits: Sequence[int],
    *,
    overlap: int,
) -> List[Tuple[int, int]]:
    boundaries = [0, *[int(s) for s in splits], int(width)]
    ranges: List[Tuple[int, int]] = []

    for index in range(len(boundaries) - 1):
        start = boundaries[index]
        end = boundaries[index + 1]

        if index > 0:
            start = max(0, start - int(overlap))

        if index < len(boundaries) - 2:
            end = min(width, end + int(overlap))

        if end <= start:
            raise RuntimeError(f"Invalid chunk range: {start}:{end}")

        ranges.append((int(start), int(end)))

    return ranges


def _provider_command_prefix(python_path: Path) -> List[str]:
    if sys.platform == "darwin" and Path("/usr/bin/arch").exists():
        return ["/usr/bin/arch", "-arm64", str(python_path)]
    return [str(python_path)]


def _run_provider(
    *,
    project_root: Path,
    workspace_base: Path,
    workspace_run_id: str,
    provider_key: str,
    device: str,
    timeout_seconds: int,
    log_path: Path,
) -> None:
    config = PROVIDERS[provider_key]
    python_path = project_root / config["python_rel"]

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
        str(workspace_base),
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
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        raise RuntimeError(
            f"{provider_key} timed out after {timeout_seconds}s. Log: {log_path}"
        ) from exc

    elapsed = time.perf_counter() - started

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (
            f"$ {' '.join(command)}\n\n"
            f"{completed.stdout or ''}\n\n"
            f"[exit_code={completed.returncode}] "
            f"[runtime_seconds={elapsed:.3f}]\n"
        ),
        encoding="utf-8",
    )

    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-30:])
        raise RuntimeError(
            f"{provider_key} failed with exit code {completed.returncode}.\n"
            f"{tail}\nLog: {log_path}"
        )


def _find_source_line(
    manifest: Dict[str, Any],
    line_id: str,
) -> Dict[str, Any]:
    for line in manifest.get("lines", []):
        if str(line.get("line_id")) == line_id:
            return dict(line)

    raise KeyError(f"{line_id!r} not found in Stage-4 manifest")


def _normalise_for_similarity(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    kept: List[str] = []

    for ch in text:
        cp = ord(ch)

        if (
            0x1CD0 <= cp <= 0x1CFF
            or 0xA8E0 <= cp <= 0xA8FF
            or 0x0300 <= cp <= 0x036F
        ):
            continue

        if ch.isspace():
            continue

        category = unicodedata.category(ch)

        if category.startswith("P") or category.startswith("C"):
            continue

        kept.append(ch)

    return "".join(kept)


def _best_window_similarity(
    output_text: str,
    reference_text: str,
) -> float | None:
    output = _normalise_for_similarity(output_text)
    reference = _normalise_for_similarity(reference_text)

    if not reference:
        return None

    if not output:
        return 0.0

    if reference in output:
        return 1.0

    ref_len = len(reference)

    min_window = max(1, int(round(ref_len * 0.70)))
    max_window = min(
        len(output),
        max(min_window, int(round(ref_len * 1.40))),
    )

    if len(output) <= max_window:
        return round(
            difflib.SequenceMatcher(None, output, reference).ratio(),
            4,
        )

    best = 0.0

    for window_len in range(min_window, max_window + 1):
        for start in range(0, len(output) - window_len + 1):
            window = output[start:start + window_len]
            score = difflib.SequenceMatcher(None, window, reference).ratio()
            if score > best:
                best = score

    return round(best, 4)


def _merge_chunk_texts(texts: Sequence[str]) -> str:
    """
    Heuristic diagnostic merger for overlapping chunk outputs.

    Exact chunk outputs remain preserved. This only creates a reconstructed
    candidate so we can compare full-line/chunked strategies consistently.
    """
    cleaned = [str(t or "").strip() for t in texts if str(t or "").strip()]

    if not cleaned:
        return ""

    merged = cleaned[0]

    for right in cleaned[1:]:
        left = merged
        max_overlap = min(40, len(left), len(right))
        best_overlap = 0
        best_score = 0.0

        for overlap in range(3, max_overlap + 1):
            left_tail = _normalise_for_similarity(left[-overlap:])
            right_head = _normalise_for_similarity(right[:overlap])

            if not left_tail or not right_head:
                continue

            score = difflib.SequenceMatcher(
                None,
                left_tail,
                right_head,
            ).ratio()

            if score > best_score:
                best_score = score
                best_overlap = overlap

        if best_score >= 0.72 and best_overlap > 0:
            merged = left + right[best_overlap:]
        else:
            merged = left.rstrip() + " " + right.lstrip()

    return " ".join(merged.split())


def _extract_provider_lines(
    manifest: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for line in manifest.get("lines", []):
        line_id = str(line.get("line_id", ""))
        if not line_id:
            continue

        quality = line.get("quality") or {}

        result[line_id] = {
            "status": line.get("status"),
            "raw_text": line.get(
                "raw_text",
                line.get("raw_iast", ""),
            ),
            "devanagari_text": line.get(
                "devanagari_text",
                line.get("devanagari", ""),
            ),
            "decoder_stability": quality.get("decoder_stability"),
            "script_purity": quality.get("script_purity"),
            "warnings": quality.get("warnings", []),
        }

    return result


def _prepare_workspace(
    *,
    source_run_dir: Path,
    diagnostic_root: Path,
    line_id: str,
    overlap_px: int,
) -> Dict[str, Any]:
    source_manifest = _load_json(
        source_run_dir / "L4" / "line_manifest.json"
    )

    source_line = _find_source_line(source_manifest, line_id)

    x = int(source_line["x"])
    y = int(source_line["y"])
    w = int(source_line["w"])
    h = int(source_line["h"])

    pad_x = int(source_line.get("htr_padding_x", 12))
    pad_y = int(
        source_line.get(
            "htr_padding_y",
            max(4, round(h * 0.12)),
        )
    )

    pages = {
        "raw": _load_image(source_run_dir / "L0" / "raw.png"),
        "balanced": _load_image(source_run_dir / "L1" / "balanced.png"),
        "binary": _load_image(source_run_dir / "L1" / "binary.png"),
    }

    shapes = {key: _image_hw(value) for key, value in pages.items()}

    if len(set(shapes.values())) != 1:
        raise ValueError(f"Page dimensions differ: {shapes}")

    cores = {
        view: _crop(pages[view], x, y, w, h)
        for view in ("raw", "balanced")
    }

    binary_core = _crop(pages["binary"], x, y, w, h)

    split_map = {
        "full": [],
        "2chunk": _discover_splits(binary_core, 2),
        "3chunk": _discover_splits(binary_core, 3),
    }

    range_map = {
        "full": [(0, w)],
        "2chunk": _chunk_ranges(
            w,
            split_map["2chunk"],
            overlap=overlap_px,
        ),
        "3chunk": _chunk_ranges(
            w,
            split_map["3chunk"],
            overlap=overlap_px,
        ),
    }

    workspace_base = diagnostic_root / "workspace"
    workspace_run_id = f"{line_id}_chunking"
    workspace_run_dir = workspace_base / workspace_run_id

    if workspace_run_dir.exists():
        shutil.rmtree(workspace_run_dir)

    crop_dir = workspace_run_dir / "L4" / "lines"
    crop_dir.mkdir(parents=True, exist_ok=True)

    inputs_dir = diagnostic_root / "inputs"

    if inputs_dir.exists():
        shutil.rmtree(inputs_dir)

    inputs_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_lines: List[Dict[str, Any]] = []
    input_records: Dict[str, Any] = {}

    reading_order = 0

    for view in VIEWS:
        input_records[view] = {}

        for variant in VARIANTS:
            input_records[view][variant] = []

            ranges = range_map[variant]

            for chunk_index, (start, end) in enumerate(ranges, start=1):
                reading_order += 1

                core_chunk = cores[view][:, start:end].copy()
                htr_crop = _pad_for_htr(
                    core_chunk,
                    pad_x=pad_x,
                    pad_y=pad_y,
                )

                diagnostic_line_id = (
                    f"{view}_{variant}_{chunk_index:02d}"
                )

                rel_path = f"L4/lines/{diagnostic_line_id}.png"
                workspace_path = workspace_run_dir / rel_path

                if not cv2.imwrite(str(workspace_path), htr_crop):
                    raise RuntimeError(
                        f"Failed to write diagnostic crop: {workspace_path}"
                    )

                friendly_path = (
                    inputs_dir
                    / f"{diagnostic_line_id}.png"
                )

                if not cv2.imwrite(str(friendly_path), htr_crop):
                    raise RuntimeError(
                        f"Failed to write diagnostic input: {friendly_path}"
                    )

                line_record = copy.deepcopy(source_line)
                line_record["line_id"] = diagnostic_line_id
                line_record["reading_order"] = reading_order
                line_record["crop_rel_path"] = rel_path
                line_record["source_line_id"] = line_id
                line_record["ablation_view"] = view
                line_record["ablation_variant"] = variant
                line_record["chunk_index"] = chunk_index
                line_record["chunk_count"] = len(ranges)
                line_record["chunk_x_start_in_line"] = int(start)
                line_record["chunk_x_end_in_line"] = int(end)
                line_record["htr_crop_width"] = int(htr_crop.shape[1])
                line_record["htr_crop_height"] = int(htr_crop.shape[0])

                diagnostic_lines.append(line_record)

                input_records[view][variant].append(
                    {
                        "line_id": diagnostic_line_id,
                        "chunk_index": chunk_index,
                        "chunk_count": len(ranges),
                        "x_start_in_line": int(start),
                        "x_end_in_line": int(end),
                        "width": int(end - start),
                        "input_path": str(friendly_path),
                    }
                )

    workspace_manifest = copy.deepcopy(source_manifest)
    workspace_manifest["stage"] = "stage4_chunking_diagnostic_input"
    workspace_manifest["run_id"] = workspace_run_id
    workspace_manifest["lines"] = diagnostic_lines

    metrics = dict(workspace_manifest.get("metrics", {}))
    metrics["num_lines"] = len(diagnostic_lines)
    metrics["diagnostic_source_run"] = source_run_dir.name
    metrics["diagnostic_source_line"] = line_id
    metrics["diagnostic_views"] = list(VIEWS)
    metrics["diagnostic_variants"] = list(VARIANTS)

    workspace_manifest["metrics"] = metrics

    workspace_manifest["diagnostic"] = {
        "version": DIAGNOSTIC_VERSION,
        "source_run": source_run_dir.name,
        "source_line": line_id,
        "views": list(VIEWS),
        "variants": list(VARIANTS),
        "split_points": split_map,
        "ranges": {
            key: [[int(a), int(b)] for a, b in value]
            for key, value in range_map.items()
        },
        "overlap_px": int(overlap_px),
        "same_vertical_geometry": True,
        "split_discovery_source": "Stage-1 binary column-ink profile",
        "note": (
            "RAW/BALANCED recognition crops use the same Stage-4 physical-line "
            "geometry. Binary is used only to locate low-ink horizontal split points."
        ),
    }

    _write_json(
        workspace_run_dir / "L4" / "line_manifest.json",
        workspace_manifest,
    )

    return {
        "workspace_base": workspace_base,
        "workspace_run_id": workspace_run_id,
        "workspace_run_dir": workspace_run_dir,
        "source_line": source_line,
        "source_manifest_version": source_manifest.get("version"),
        "split_points": split_map,
        "ranges": range_map,
        "inputs": input_records,
    }


def _build_results(
    *,
    provider_lines: Dict[str, Dict[str, Dict[str, Any]]],
    reference_prefix: str,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for view in VIEWS:
        results[view] = {}

        for variant in VARIANTS:
            results[view][variant] = {}

            chunk_count = {
                "full": 1,
                "2chunk": 2,
                "3chunk": 3,
            }[variant]

            line_ids = [
                f"{view}_{variant}_{index:02d}"
                for index in range(1, chunk_count + 1)
            ]

            for provider_key in ("provider_a", "provider_b"):
                chunks = [
                    dict(provider_lines[provider_key][line_id])
                    for line_id in line_ids
                ]

                reconstructed = _merge_chunk_texts(
                    [
                        chunk.get("devanagari_text", "")
                        for chunk in chunks
                    ]
                )

                results[view][variant][provider_key] = {
                    "provider": PROVIDERS[provider_key]["provider"],
                    "chunks": chunks,
                    "reconstructed_devanagari": reconstructed,
                    "reference_best_window_similarity": (
                        _best_window_similarity(
                            reconstructed,
                            reference_prefix,
                        )
                        if reference_prefix
                        else None
                    ),
                }

    return results


def _best_configuration(
    results: Dict[str, Any],
) -> Dict[str, Any] | None:
    candidates = []

    for view in VIEWS:
        for variant in VARIANTS:
            for provider_key in ("provider_a", "provider_b"):
                score = (
                    results[view][variant][provider_key]
                    .get("reference_best_window_similarity")
                )

                if score is None:
                    continue

                candidates.append(
                    {
                        "view": view,
                        "variant": variant,
                        "provider_key": provider_key,
                        "provider": PROVIDERS[provider_key]["provider"],
                        "score": float(score),
                    }
                )

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["score"])


def _print_results(output: Dict[str, Any]) -> None:
    print()
    print("=" * 105)
    print("STAGE-5 LONG-LINE CHUNKING DIAGNOSTIC")
    print("=" * 105)

    print("Source run:", output["source_run_id"])
    print("Source line:", output["source_line_id"])
    print("Source Stage-4 version:", output["source_stage4_version"])
    print("Source line width:", output["source_line_width"])
    print("Source line height:", output["source_line_height"])
    print("Aspect ratio:", output["source_line_aspect_ratio"])

    print()
    print("Split geometry:")
    print("  2-chunk split:", output["split_points"]["2chunk"])
    print("  3-chunk splits:", output["split_points"]["3chunk"])

    if output.get("reference_prefix"):
        print()
        print("Diagnostic reference prefix:")
        print(output["reference_prefix"])

    for view in VIEWS:
        for variant in VARIANTS:
            print()
            print("-" * 105)
            print(f"{view.upper()} / {variant.upper()}")

            for provider_key in ("provider_a", "provider_b"):
                item = output["results"][view][variant][provider_key]

                print()
                print(
                    f"{provider_key.upper()} "
                    f"({item['provider']})"
                )
                print("Reconstructed:")
                print(item["reconstructed_devanagari"] or "<blank>")

                if output.get("reference_prefix"):
                    print(
                        "reference_best_window_similarity:",
                        item["reference_best_window_similarity"],
                    )

                if len(item["chunks"]) > 1:
                    for index, chunk in enumerate(item["chunks"], start=1):
                        print(f"  chunk {index}: {chunk.get('devanagari_text', '')}")

    best = output.get("best_configuration")

    print()
    print("=" * 105)
    print("BEST DIAGNOSTIC CONFIGURATION")
    print("=" * 105)

    if best:
        print(
            f"{best['view']} / {best['variant']} / {best['provider']} "
            f"-> similarity {best['score']}"
        )
    else:
        print("No diagnostic reference was supplied.")

    print()
    print("Interpretation:")
    print("  Chunking substantially improves similarity -> Stage-5 input granularity is a major factor.")
    print("  Full line remains best -> chunking is not the main issue.")
    print("  All configurations remain poor -> Provider A/B manuscript-domain mismatch dominates.")
    print("  RAW consistently beats BALANCED -> revisit Stage-1 HTR input selection.")
    print("  BALANCED consistently beats RAW -> keep balanced as the preferred HTR view.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether long-line aspect ratio is hurting Stage-5 HTR by "
            "comparing full-line, 2-chunk and 3-chunk recognition on RAW and "
            "BALANCED versions of the same corrected Stage-4 physical line."
        )
    )

    parser.add_argument("run_id")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--line-id", default="line_001")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument(
        "--reference-prefix",
        default="",
        help=(
            "Optional known visible prefix used only for diagnostic similarity. "
            "It is not stored as scholar-certified ground truth."
        ),
    )
    parser.add_argument(
        "--overlap-px",
        type=int,
        default=28,
        help="Horizontal overlap added on both sides of internal chunk boundaries.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
    )

    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    artifacts_path = Path(args.artifacts)

    if not artifacts_path.is_absolute():
        artifacts_path = project_root / artifacts_path

    artifacts_path = artifacts_path.resolve()
    source_run_dir = artifacts_path / args.run_id

    if not source_run_dir.is_dir():
        raise FileNotFoundError(f"Run does not exist: {source_run_dir}")

    diagnostic_root = (
        source_run_dir
        / "stage5_chunking_diagnostic"
    )

    diagnostic_root.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_workspace(
        source_run_dir=source_run_dir,
        diagnostic_root=diagnostic_root,
        line_id=args.line_id,
        overlap_px=args.overlap_px,
    )

    logs_dir = diagnostic_root / "logs"
    started = time.time()

    for provider_key in ("provider_a", "provider_b"):
        _run_provider(
            project_root=project_root,
            workspace_base=Path(prepared["workspace_base"]),
            workspace_run_id=str(prepared["workspace_run_id"]),
            provider_key=provider_key,
            device=args.device,
            timeout_seconds=args.timeout_seconds,
            log_path=logs_dir / f"{provider_key}.log",
        )

    workspace_run_dir = Path(prepared["workspace_run_dir"])

    provider_lines: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for provider_key in ("provider_a", "provider_b"):
        manifest_path = (
            workspace_run_dir
            / PROVIDERS[provider_key]["snapshot"]
            / "htr_manifest.json"
        )

        provider_manifest = _load_json(manifest_path)
        provider_lines[provider_key] = _extract_provider_lines(provider_manifest)

    results = _build_results(
        provider_lines=provider_lines,
        reference_prefix=args.reference_prefix,
    )

    source_line = prepared["source_line"]
    source_width = int(source_line["w"])
    source_height = int(source_line["h"])

    output: Dict[str, Any] = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "source_run_id": args.run_id,
        "source_stage4_version": prepared["source_manifest_version"],
        "source_line_id": args.line_id,
        "source_line_width": source_width,
        "source_line_height": source_height,
        "source_line_aspect_ratio": round(
            source_width / float(max(1, source_height)),
            4,
        ),
        "reference_prefix": args.reference_prefix or None,
        "reference_note": (
            "Used only for diagnostic similarity; not automatically treated "
            "as scholar-certified ground truth."
            if args.reference_prefix
            else None
        ),
        "controlled_variables": {
            "same_stage4_physical_line": True,
            "same_vertical_geometry": True,
            "binary_used_only_for_split_discovery": True,
            "views_under_test": list(VIEWS),
            "variants_under_test": list(VARIANTS),
            "chunk_overlap_px": int(args.overlap_px),
        },
        "split_points": prepared["split_points"],
        "ranges": {
            key: [[int(a), int(b)] for a, b in value]
            for key, value in prepared["ranges"].items()
        },
        "inputs": prepared["inputs"],
        "results": results,
        "runtime_seconds": round(time.time() - started, 3),
        "artifacts": {
            "inputs_dir": str(diagnostic_root / "inputs"),
            "provider_a_log": str(logs_dir / "provider_a.log"),
            "provider_b_log": str(logs_dir / "provider_b.log"),
            "workspace": str(workspace_run_dir),
        },
    }

    output["best_configuration"] = _best_configuration(results)

    output_path = diagnostic_root / "chunking_results.json"
    _write_json(output_path, output)

    _print_results(output)

    print()
    print("Result artifact:", output_path)


if __name__ == "__main__":
    main()
