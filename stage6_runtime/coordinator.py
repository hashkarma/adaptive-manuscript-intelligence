from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


STAGE6_COORDINATOR_VERSION = "0.1.0-stage6-integrated-isolated-coordinator"


class Stage6PipelineError(RuntimeError):
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


def _tail(text: str, lines: int = 35) -> str:
    rows = (text or "").splitlines()
    return "\n".join(rows[-lines:])


def _python_prefix(python_path: Path) -> List[str]:
    """
    Force native ARM on Apple Silicon.

    This mirrors the already validated Stage-5 coordinator design and protects
    native PyTorch/Pillow/Vidyut/RapidFuzz environments from a Rosetta parent
    FastAPI process.
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


def _validate_python(path: Path, label: str) -> None:
    if not path.exists():
        raise Stage6PipelineError(
            "environment_validation",
            f"{label} Python executable is missing: {path}",
        )

    if not os.access(path, os.X_OK):
        raise Stage6PipelineError(
            "environment_validation",
            f"{label} Python is not executable: {path}",
        )


def _run_process(
    *,
    command: List[str],
    cwd: Path,
    log_path: Path,
    stage: str,
    timeout_seconds: int,
) -> float:
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
        output = exc.stdout if isinstance(exc.stdout, str) else ""

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output or "", encoding="utf-8")

        raise Stage6PipelineError(
            stage,
            f"Timed out after {timeout_seconds} seconds.",
            log_path=str(log_path),
        ) from exc

    runtime = time.perf_counter() - started

    log_text = (
        f"$ {' '.join(command)}\n\n"
        f"{completed.stdout or ''}\n\n"
        f"[exit_code={completed.returncode}] "
        f"[runtime_seconds={runtime:.3f}]\n"
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")

    if completed.returncode != 0:
        raise Stage6PipelineError(
            stage,
            (
                f"Stage-6 subprocess exited with code "
                f"{completed.returncode}.\n"
                f"{_tail(completed.stdout or '')}"
            ),
            log_path=str(log_path),
        )

    return runtime


def _preflight(
    *,
    python_path: Path,
    label: str,
    import_statement: str,
    project_root: Path,
    log_path: Path,
) -> None:
    code = (
        "import platform, sys; "
        f"{import_statement}; "
        "print('machine=' + platform.machine()); "
        "print('python=' + sys.executable); "
        f"print('{label}=ok')"
    )

    _run_process(
        command=[
            *_python_prefix(python_path),
            "-c",
            code,
        ],
        cwd=project_root,
        log_path=log_path,
        stage=f"{label}_preflight",
        timeout_seconds=90,
    )


def _required_file(path: Path, stage: str, message: str) -> None:
    if not path.exists():
        raise Stage6PipelineError(
            stage,
            f"{message}: {path}",
        )


def _substage_command(
    *,
    python_path: Path,
    run_id: str,
    substage: str,
    artifacts_path: Path,
    vidyut_data: Path,
    corpus: Path,
) -> List[str]:
    return [
        *_python_prefix(python_path),
        "-m",
        "stage6_runtime.run_substage",
        run_id,
        "--substage",
        substage,
        "--artifacts",
        str(artifacts_path),
        "--vidyut-data",
        str(vidyut_data),
        "--corpus",
        str(corpus),
    ]


def run_stage6_pipeline(
    *,
    project_root: str | Path,
    run_id: str,
    artifacts: str | Path = "artifacts",
    vidyut_data: str | Path = "models/vidyut-0.4.0",
    corpus: str | Path = "knowledge/stage6d_dcs/passages.jsonl",
    substage_timeout_seconds: int = 1800,
) -> Dict[str, Any]:
    """
    Execute Stage 6A -> 6F using isolated native-ARM environments.

    Environment ownership:
      venv-stage6:
        6A HTR evidence parser
        6B Vidyut candidate lattice
        6C morphology/grammar
        6D deterministic contextual retrieval
        6E evidence-constrained reconstruction
        6F T(p)

      venv-stage6-rag:
        6D.2 RapidFuzz noisy-surface retrieval

    Stage 6G is intentionally NOT executed here. It remains in the main
    FastAPI/orchestration process, just as Stage 5G does for Stage 5.
    """
    project_root = Path(project_root).resolve()

    artifacts_path = Path(artifacts)
    if not artifacts_path.is_absolute():
        artifacts_path = project_root / artifacts_path
    artifacts_path = artifacts_path.resolve()

    vidyut_data_path = Path(vidyut_data)
    if not vidyut_data_path.is_absolute():
        vidyut_data_path = project_root / vidyut_data_path
    vidyut_data_path = vidyut_data_path.resolve()

    corpus_path = Path(corpus)
    if not corpus_path.is_absolute():
        corpus_path = project_root / corpus_path
    corpus_path = corpus_path.resolve()

    run_dir = artifacts_path / run_id

    _required_file(
        run_dir / "L5_readiness" / "htr_readiness.json",
        "stage5_validation",
        "Stage 5 HTR readiness is missing",
    )
    _required_file(
        run_dir / "L5_provider_A" / "htr_manifest.json",
        "stage5_validation",
        "Provider-A HTR manifest is missing",
    )
    _required_file(
        run_dir / "L5_provider_B" / "htr_manifest.json",
        "stage5_validation",
        "Provider-B HTR manifest is missing",
    )
    _required_file(
        run_dir / "L5_compare" / "provider_comparison.json",
        "stage5_validation",
        "Cross-provider comparison is missing",
    )

    _required_file(
        corpus_path,
        "corpus_validation",
        "Stage-6 retrieval corpus is missing",
    )

    if not vidyut_data_path.is_dir():
        raise Stage6PipelineError(
            "vidyut_validation",
            f"Vidyut data directory is missing: {vidyut_data_path}",
        )

    stage6_python = (
        project_root
        / "venv-stage6"
        / "bin"
        / "python"
    )
    stage6_rag_python = (
        project_root
        / "venv-stage6-rag"
        / "bin"
        / "python"
    )

    _validate_python(
        stage6_python,
        "Stage-6/Vidyut",
    )
    _validate_python(
        stage6_rag_python,
        "Stage-6/RAG",
    )

    runtime_dir = run_dir / "stage6_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    _preflight(
        python_path=stage6_python,
        label="stage6_vidyut",
        import_statement=(
            "from vidyut import lipi; "
            "from layers.layer6_semantic import "
            "run_layer6_htr_evidence_parser"
        ),
        project_root=project_root,
        log_path=runtime_dir / "stage6_vidyut_preflight.log",
    )

    _preflight(
        python_path=stage6_rag_python,
        label="stage6_rag",
        import_statement=(
            "import rapidfuzz; "
            "from layers.layer6d2_noisy_retrieval import "
            "run_layer6d2_noisy_surface_retrieval"
        ),
        project_root=project_root,
        log_path=runtime_dir / "stage6_rag_preflight.log",
    )

    started = time.time()

    plan = [
        ("6a", stage6_python),
        ("6b", stage6_python),
        ("6c", stage6_python),
        ("6d", stage6_python),
        ("6d2", stage6_rag_python),
        ("6e", stage6_python),
        ("6f", stage6_python),
    ]

    substages: List[Dict[str, Any]] = []

    for substage, python_path in plan:
        log_path = runtime_dir / f"substage_{substage}.log"

        runtime = _run_process(
            command=_substage_command(
                python_path=python_path,
                run_id=run_id,
                substage=substage,
                artifacts_path=artifacts_path,
                vidyut_data=vidyut_data_path,
                corpus=corpus_path,
            ),
            cwd=project_root,
            log_path=log_path,
            stage=f"stage6_{substage}",
            timeout_seconds=substage_timeout_seconds,
        )

        summary_path = (
            runtime_dir
            / f"substage_{substage}_summary.json"
        )

        summary = _load_json(summary_path)
        summary["log_path"] = str(log_path)
        summary["runtime_seconds_observed_by_coordinator"] = round(
            runtime,
            3,
        )

        substages.append(summary)

    stage6e_path = run_dir / "L6" / "stage6e_manifest.json"
    stage6f_path = run_dir / "L6" / "stage6f_manifest.json"

    _required_file(
        stage6e_path,
        "stage6e_validation",
        "Stage 6E reconstruction artifact was not produced",
    )
    _required_file(
        stage6f_path,
        "stage6f_validation",
        "Stage 6F trust artifact was not produced",
    )

    stage6e = _load_json(stage6e_path)
    stage6f = _load_json(stage6f_path)

    summary = {
        "coordinator_version": STAGE6_COORDINATOR_VERSION,
        "run_id": run_id,
        "execution_mode": "integrated_isolated_subprocesses",
        "host_machine": platform.machine(),
        "environments": {
            "stage6_vidyut": str(stage6_python),
            "stage6_rag": str(stage6_rag_python),
        },
        "substage_plan": [
            row[0]
            for row in plan
        ],
        "substages": substages,
        "reconstruction": {
            "stage6e_version": stage6e.get("version"),
            "metrics": stage6e.get("metrics", {}),
        },
        "trust": {
            "stage6f_version": stage6f.get("version"),
            "metrics": stage6f.get("metrics", {}),
        },
        "artifacts": {
            "stage6a": str(
                run_dir / "L6" / "stage6_manifest.json"
            ),
            "stage6b": str(
                run_dir / "L6" / "stage6b_manifest.json"
            ),
            "stage6c": str(
                run_dir / "L6" / "stage6c_manifest.json"
            ),
            "stage6d": str(
                run_dir / "L6" / "stage6d_manifest.json"
            ),
            "stage6d2": str(
                run_dir / "L6" / "stage6d2_manifest.json"
            ),
            "stage6e": str(stage6e_path),
            "stage6f": str(stage6f_path),
        },
        "runtime_seconds": round(
            time.time() - started,
            3,
        ),
        "orchestration_pending": True,
        "next_action": "run_stage6g_adaptive_retry_controller",
        "note": (
            "Stage 6A-6F completed in isolated native-ARM environments. "
            "Stage 6G routing remains owned by the main orchestration evaluator."
        ),
    }

    summary_path = runtime_dir / "stage6_pipeline_summary.json"
    _write_json(summary_path, summary)

    return summary
