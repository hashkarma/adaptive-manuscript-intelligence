from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from compare_stage5_providers import compare_manifests
from evaluate_stage5_readiness import evaluate_readiness


STAGE5_COORDINATOR_VERSION = "0.1.2-stage5-integrated-coordinator"


class Stage5PipelineError(RuntimeError):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        log_path: str | None = None,
    ) -> None:
        self.stage = stage
        self.log_path = log_path

        detail = f"{stage}: {message}"

        if log_path:
            detail += f" Log: {log_path}"

        super().__init__(detail)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _write_json(
    path: Path,
    data: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def _tail(
    text: str,
    lines: int = 25,
) -> str:
    rows = (text or "").splitlines()
    return "\n".join(rows[-lines:])


def _run_subprocess(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
    stage: str,
    timeout_seconds: int,
) -> None:
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=os.environ.copy(),
        )

    except subprocess.TimeoutExpired as exc:
        output = (
            (exc.stdout or "")
            if isinstance(exc.stdout, str)
            else ""
        )

        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        log_path.write_text(
            output,
            encoding="utf-8",
        )

        raise Stage5PipelineError(
            stage,
            f"Timed out after {timeout_seconds} seconds.",
            log_path=str(log_path),
        ) from exc

    duration = time.perf_counter() - started

    log_text = (
        f"$ {' '.join(command)}\n\n"
        f"{completed.stdout or ''}\n\n"
        f"[exit_code={completed.returncode}] "
        f"[runtime_seconds={duration:.3f}]\n"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_path.write_text(
        log_text,
        encoding="utf-8",
    )

    if completed.returncode != 0:
        raise Stage5PipelineError(
            stage,
            (
                f"Provider process exited with code "
                f"{completed.returncode}.\n"
                f"{_tail(completed.stdout or '')}"
            ),
            log_path=str(log_path),
        )


def _promote_snapshot(
    *,
    run_dir: Path,
    temporary_name: str,
    canonical_name: str,
) -> None:
    temporary = run_dir / temporary_name
    canonical = run_dir / canonical_name
    previous = run_dir / f"{canonical_name}_previous"

    if not temporary.is_dir():
        raise FileNotFoundError(
            f"Temporary provider snapshot is missing: {temporary}"
        )

    if previous.exists():
        shutil.rmtree(previous)

    if canonical.exists():
        canonical.rename(previous)

    temporary.rename(canonical)


def _provider_python_prefix(
    python_path: Path,
) -> list[str]:
    """
    Return the command prefix used to launch an isolated provider Python.

    On Apple Silicon, the main FastAPI environment may itself be running under
    Rosetta/x86_64.  A universal Python launched as its child can then inherit
    the x86_64 slice even though the provider environment contains native
    arm64 wheels (Pillow, PyTorch, etc.).  Force arm64 for the provider
    subprocesses so their interpreter and compiled wheels use the same
    architecture.

    This does not alter either virtual environment.
    """
    if sys.platform == "darwin":
        arch_tool = Path("/usr/bin/arch")

        if arch_tool.exists():
            return [
                str(arch_tool),
                "-arm64",
                str(python_path),
            ]

    return [str(python_path)]


def _provider_architecture_preflight(
    *,
    python_path: Path,
    label: str,
    project_root: Path,
    log_path: Path,
    timeout_seconds: int = 60,
) -> None:
    """
    Validate that the provider interpreter starts in the expected architecture
    and can import Pillow before full model inference begins.
    """
    command = [
        *_provider_python_prefix(python_path),
        "-c",
        (
            "import platform, sys; "
            "from PIL import Image; "
            "print('machine=' + platform.machine()); "
            "print('python=' + sys.executable); "
            "print('pillow=ok')"
        ),
    ]

    _run_subprocess(
        command=command,
        cwd=project_root,
        log_path=log_path,
        stage=f"{label}_architecture_preflight",
        timeout_seconds=timeout_seconds,
    )


def _repair_promoted_runtime_summary(
    *,
    run_dir: Path,
    canonical_name: str,
) -> Dict[str, Any]:
    """
    After a validated *_next snapshot is promoted, rewrite its runtime summary
    so audit metadata points to the canonical directory that actually exists.
    """
    snapshot_dir = run_dir / canonical_name
    summary_path = snapshot_dir / "runtime_summary.json"
    manifest_path = snapshot_dir / "htr_manifest.json"

    summary = _load_json(summary_path)

    summary["snapshot_name"] = canonical_name
    summary["snapshot_path"] = str(snapshot_dir)
    summary["manifest_path"] = str(manifest_path)

    _write_json(
        summary_path,
        summary,
    )

    return summary


def _validate_environment_python(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise Stage5PipelineError(
            "environment_validation",
            f"{label} Python executable is missing: {path}",
        )

    if not os.access(
        path,
        os.X_OK,
    ):
        raise Stage5PipelineError(
            "environment_validation",
            f"{label} Python is not executable: {path}",
        )


def _stage4_S(
    run_dir: Path,
) -> float:
    manifest = _load_json(
        run_dir
        / "L4"
        / "line_manifest.json"
    )

    metrics = manifest.get(
        "metrics",
        {},
    )

    value = metrics.get(
        "segmentation_confidence"
    )

    if value is None:
        raise Stage5PipelineError(
            "readiness",
            "Stage 4 segmentation_confidence is unavailable.",
        )

    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def run_stage5_pipeline(
    *,
    project_root: str | Path,
    run_id: str,
    artifacts: str | Path = "artifacts",
    device: str = "auto",
    num_beams: int = 4,
    n_best: int = 3,
    max_output_length: int = 192,
    provider_timeout_seconds: int = 1800,
) -> Dict[str, Any]:
    """
    Execute Research Stage 5 end to end.

    The coordinator itself runs in the main application environment.
    Provider A and Provider B inference run in isolated virtual environments.

    Pipeline:
      1. validate Stage 4
      2. Provider A inference
      3. Provider B inference
      4. promote fresh provider snapshots
      5. A/B comparison
      6. H(p) readiness evaluation

    Stage 5G orchestration remains owned by orchestration/evaluator.py in the
    FastAPI process after this function returns.
    """
    project_root = Path(
        project_root
    ).resolve()

    artifacts_path = Path(
        artifacts
    )

    if not artifacts_path.is_absolute():
        artifacts_path = (
            project_root
            / artifacts_path
        )

    artifacts_path = artifacts_path.resolve()

    run_dir = (
        artifacts_path
        / run_id
    )

    stage4_manifest = (
        run_dir
        / "L4"
        / "line_manifest.json"
    )

    if not stage4_manifest.exists():
        raise Stage5PipelineError(
            "stage4_validation",
            (
                "Stage 4 line_manifest.json is missing. "
                "Complete Stage 4 before HTR."
            ),
        )

    provider_a_python = (
        project_root
        / "venv-stage5"
        / "bin"
        / "python"
    )

    provider_b_python = (
        project_root
        / "venv-provider-b"
        / "bin"
        / "python"
    )

    _validate_environment_python(
        provider_a_python,
        "Provider A",
    )

    _validate_environment_python(
        provider_b_python,
        "Provider B",
    )

    runtime_dir = (
        run_dir
        / "stage5_runtime"
    )

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _provider_architecture_preflight(
        python_path=provider_a_python,
        label="provider_a",
        project_root=project_root,
        log_path=runtime_dir / "provider_a_preflight.log",
    )

    _provider_architecture_preflight(
        python_path=provider_b_python,
        label="provider_b",
        project_root=project_root,
        log_path=runtime_dir / "provider_b_preflight.log",
    )

    started_at = time.time()

    temp_a = "L5_provider_A_next"
    temp_b = "L5_provider_B_next"

    for stale_name in [
        temp_a,
        temp_b,
    ]:
        stale = run_dir / stale_name

        if stale.exists():
            shutil.rmtree(stale)

    common = [
        "--artifacts",
        str(artifacts_path),
        "--device",
        device,
        "--num-beams",
        str(num_beams),
        "--n-best",
        str(n_best),
        "--max-output-length",
        str(max_output_length),
    ]

    command_a = [
        *_provider_python_prefix(provider_a_python),
        "-m",
        "stage5_runtime.run_provider",
        run_id,
        "--provider",
        "trocr_iast_baseline",
        "--snapshot",
        temp_a,
        *common,
    ]

    command_b = [
        *_provider_python_prefix(provider_b_python),
        "-m",
        "stage5_runtime.run_provider",
        run_id,
        "--provider",
        "trocr_vedic_devanagari",
        "--snapshot",
        temp_b,
        *common,
    ]

    _run_subprocess(
        command=command_a,
        cwd=project_root,
        log_path=(
            runtime_dir
            / "provider_a.log"
        ),
        stage="provider_a",
        timeout_seconds=provider_timeout_seconds,
    )

    _run_subprocess(
        command=command_b,
        cwd=project_root,
        log_path=(
            runtime_dir
            / "provider_b.log"
        ),
        stage="provider_b",
        timeout_seconds=provider_timeout_seconds,
    )

    # Validate temporary manifests before replacing the last known-good
    # canonical snapshots.
    manifest_a_next = (
        run_dir
        / temp_a
        / "htr_manifest.json"
    )

    manifest_b_next = (
        run_dir
        / temp_b
        / "htr_manifest.json"
    )

    if not manifest_a_next.exists():
        raise Stage5PipelineError(
            "provider_a",
            f"Fresh Provider-A manifest is missing: {manifest_a_next}",
        )

    if not manifest_b_next.exists():
        raise Stage5PipelineError(
            "provider_b",
            f"Fresh Provider-B manifest is missing: {manifest_b_next}",
        )

    _promote_snapshot(
        run_dir=run_dir,
        temporary_name=temp_a,
        canonical_name="L5_provider_A",
    )

    _promote_snapshot(
        run_dir=run_dir,
        temporary_name=temp_b,
        canonical_name="L5_provider_B",
    )

    manifest_a_path = (
        run_dir
        / "L5_provider_A"
        / "htr_manifest.json"
    )

    manifest_b_path = (
        run_dir
        / "L5_provider_B"
        / "htr_manifest.json"
    )

    manifest_a = _load_json(
        manifest_a_path
    )

    manifest_b = _load_json(
        manifest_b_path
    )

    comparison = compare_manifests(
        manifest_a,
        manifest_b,
    )

    comparison_path = (
        run_dir
        / "L5_compare"
        / "provider_comparison.json"
    )

    _write_json(
        comparison_path,
        comparison,
    )

    readiness = evaluate_readiness(
        stage4_S=_stage4_S(run_dir),
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        comparison=comparison,
    )

    readiness_path = (
        run_dir
        / "L5_readiness"
        / "htr_readiness.json"
    )

    _write_json(
        readiness_path,
        readiness,
    )

    provider_a_summary = _repair_promoted_runtime_summary(
        run_dir=run_dir,
        canonical_name="L5_provider_A",
    )

    provider_b_summary = _repair_promoted_runtime_summary(
        run_dir=run_dir,
        canonical_name="L5_provider_B",
    )

    page = readiness.get(
        "page",
        {},
    )

    comparison_aggregate = comparison.get(
        "aggregate",
        {},
    )

    summary = {
        "coordinator_version": STAGE5_COORDINATOR_VERSION,
        "run_id": run_id,
        "execution_mode": "integrated_isolated_subprocesses",
        "provider_a": provider_a_summary,
        "provider_b": provider_b_summary,
        "comparison": {
            "mean_content_char_similarity": (
                comparison_aggregate.get(
                    "mean_content_char_similarity"
                )
            ),
            "median_content_char_similarity": (
                comparison_aggregate.get(
                    "median_content_char_similarity"
                )
            ),
            "low_agreement_lines": (
                comparison_aggregate.get(
                    "low_agreement_lines"
                )
            ),
        },
        "readiness": {
            "H": page.get(
                "htr_readiness_H_page"
            ),
            "evidence": page.get(
                "evidence",
                {},
            ),
            "low_readiness_lines": page.get(
                "low_readiness_lines"
            ),
            "low_agreement_lines": page.get(
                "low_agreement_lines"
            ),
            "cer": page.get("cer"),
            "wer": page.get("wer"),
            "ground_truth_available": page.get(
                "ground_truth_available",
                False,
            ),
        },
        "artifacts": {
            "provider_a_manifest": str(
                manifest_a_path
            ),
            "provider_b_manifest": str(
                manifest_b_path
            ),
            "provider_comparison": str(
                comparison_path
            ),
            "htr_readiness": str(
                readiness_path
            ),
            "provider_a_log": str(
                runtime_dir / "provider_a.log"
            ),
            "provider_b_log": str(
                runtime_dir / "provider_b.log"
            ),
        },
        "runtime_seconds": round(
            time.time() - started_at,
            3,
        ),
        "orchestration_pending": True,
        "note": (
            "Provider inference, cross-provider comparison and H(p) are "
            "complete. Stage 5G routing should now be executed by the "
            "main orchestration evaluator."
        ),
    }

    summary_path = (
        runtime_dir
        / "stage5_pipeline_summary.json"
    )

    _write_json(
        summary_path,
        summary,
    )

    return summary
