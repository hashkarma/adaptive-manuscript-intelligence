from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import unicodedata
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.artifact_store import ArtifactStore
from core.io_utils import read_image_bgr
from layers.layer0_ingest import run_layer0_ingest
from layers.layer1_restore import run_layer1_restore
from layers.layer2_damage import run_layer2_damage
from layers.layer3_layout import run_layer3_layout
from layers.layer4_segment import run_layer4_segmentation
from orchestration.evaluator import (
    evaluate_and_save_layer1,
    evaluate_and_save_layer2,
    evaluate_and_save_layer3,
    evaluate_and_save_layer4,
    evaluate_and_save_layer5,
    evaluate_and_save_layer6,
    finalize_orchestration,
)
from stage5_runtime.coordinator import (
    Stage5PipelineError,
    run_stage5_pipeline,
    run_stage5_provider_c_retry,
)
from stage6_runtime.coordinator import (
    Stage6PipelineError,
    run_stage6_pipeline,
)

from ground_truth_runtime.evaluator import (
    GroundTruthValidationError,
    attach_and_evaluate_ground_truth,
    load_ground_truth_bundle,
)


app = FastAPI(title="Manuscript Intelligence Platform")

# === SHOWCASE DASHBOARD BEGIN ===
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

SHOWCASE_FRONTEND_DIR = (Path(__file__).resolve().parents[1] / "frontend").resolve()

app.mount(
    "/showcase-assets",
    StaticFiles(directory=str(SHOWCASE_FRONTEND_DIR)),
    name="showcase-assets",
)

@app.get("/showcase", include_in_schema=False)
def showcase_dashboard():
    return FileResponse(str(SHOWCASE_FRONTEND_DIR / "showcase.html"))
# === SHOWCASE DASHBOARD END ===


# === SHOWCASE API BEGIN ===
from showcase.api.showcase_router import router as showcase_router
app.include_router(showcase_router)
# === SHOWCASE API END ===


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE5_ACTIVE_RUNS: set[str] = set()
STAGE5_ACTIVE_RUNS_LOCK = threading.Lock()

STAGE6_ACTIVE_RUNS: set[str] = set()
STAGE6_ACTIVE_RUNS_LOCK = threading.Lock()


# GENERIC_PLATFORM_ROUTE_V1
@app.get("/platform", include_in_schema=False)
def platform_page():
    return FileResponse("frontend/platform.html")


@app.get("/")
def homepage():
    return FileResponse("frontend/index.html")


@app.get("/orchestration")
def orchestration_page():
    return FileResponse("frontend/orchestration.html")


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def save_stage_state(store: ArtifactStore, stage: str, data: dict):
    store.write_json(f"state/{stage}.json", data)


def load_stage_state(store: ArtifactStore, stage: str) -> dict:
    path = store.path(f"state/{stage}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# PLATFORM_EXECUTION_PROFILE_WIRING_V1
PLATFORM_EXECUTION_PROFILE_CONFIGS = {
    "generic_default": None,
    "demo_balanced": Path("showcase/config/demo_balanced_profile.json"),
}


def normalize_execution_profile(profile_id: str | None) -> str:
    value = str(profile_id or "generic_default").strip().lower()
    if value not in PLATFORM_EXECUTION_PROFILE_CONFIGS:
        raise ValueError(
            "Unsupported execution profile. Allowed values: "
            + ", ".join(sorted(PLATFORM_EXECUTION_PROFILE_CONFIGS))
        )
    return value


def load_execution_profile_config(profile_id: str) -> dict:
    profile_id = normalize_execution_profile(profile_id)
    path = PLATFORM_EXECUTION_PROFILE_CONFIGS[profile_id]
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(
            f"Execution profile configuration is missing: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Execution profile must be a JSON object: {path}")
    return data


def stage4_parameters_for_execution_profile(profile_id: str) -> dict:
    profile_id = normalize_execution_profile(profile_id)
    profile = load_execution_profile_config(profile_id)
    if not profile:
        return {}
    params = (
        profile.get("stage4", {})
        .get("accepted_showcase_parameters", {})
    )
    if not isinstance(params, dict):
        raise ValueError(
            "stage4.accepted_showcase_parameters must be a JSON object"
        )
    return dict(params)


# STAGE6_SEMANTIC_ASSIST_RUNTIME_V1
def run_stage6_semantic_assist_subprocess(
    run_id: str,
    execution_profile: str,
    profile_config: dict,
) -> dict:
    stage6_cfg = profile_config.get("stage6", {}) or {}
    enabled = bool(
        stage6_cfg.get("qwen_semantic_assist_enabled", False)
    )

    summary = {
        "enabled": enabled,
        "executed": False,
        "status": "disabled" if not enabled else "not_run",
        "candidate_output_status": (
            "AI CANDIDATE — SCHOLAR VALIDATION PENDING"
            if enabled
            else None
        ),
        "changes_S_H_T": False,
    }

    if not enabled:
        return summary

    profile_path = PLATFORM_EXECUTION_PROFILE_CONFIGS.get(
        execution_profile
    )
    if profile_path is None:
        summary["status"] = "profile_not_available"
        return summary

    run_dir = PROJECT_ROOT / "artifacts" / run_id
    runtime_script = (
        PROJECT_ROOT
        / "core"
        / "stage6_semantic_assist_runtime.py"
    )

    cmd = [
        sys.executable,
        str(runtime_script),
        "--run-dir",
        str(run_dir),
        "--profile",
        str(PROJECT_ROOT / profile_path),
    ]

    summary["executed"] = True

    try:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        summary.update(
            {
                "status": "timeout",
                "error": "semantic_assist_timeout",
                "runtime_timeout_seconds": 900,
                "stderr": str(exc),
            }
        )
        return summary

    summary["return_code"] = completed.returncode
    summary["stdout_tail"] = completed.stdout[-2000:]
    summary["stderr_tail"] = completed.stderr[-2000:]

    audit_path = (
        run_dir
        / "L6_semantic_assist"
        / "stage6_semantic_audit.json"
    )
    candidate_path = (
        run_dir
        / "L6_semantic_assist"
        / "stage6_semantic_candidate.json"
    )

    if audit_path.exists():
        audit = load_json_if_exists(str(audit_path))
        summary["audit"] = audit
        summary["validation"] = audit.get("validation")
        summary["scientific_status"] = audit.get(
            "scientific_status"
        )

    summary["candidate_artifact"] = (
        str(candidate_path.relative_to(PROJECT_ROOT))
        if candidate_path.exists()
        else None
    )
    summary["audit_artifact"] = (
        str(audit_path.relative_to(PROJECT_ROOT))
        if audit_path.exists()
        else None
    )

    summary["status"] = (
        "ok"
        if completed.returncode == 0 and candidate_path.exists()
        else "error"
    )
    return summary


def load_json_if_exists(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def current_layer_reports(
    store: ArtifactStore,
    *,
    include_layer4: bool = True,
    include_layer5: bool = False,
    include_layer6: bool = False,
) -> dict:
    """
    Load persisted orchestration reports.

    Stage 5 is opt-in so a Stage 4 rerun cannot accidentally reuse an older
    Layer-5 report and jump directly to a stale post_stage5 decision.
    """
    reports = {
        "layer1": load_json_if_exists(
            store.path("orchestration/layer1_report.json")
        ),
        "layer2": load_json_if_exists(
            store.path("orchestration/layer2_report.json")
        ),
        "layer3": load_json_if_exists(
            store.path("orchestration/layer3_report.json")
        ),
    }

    if include_layer4:
        reports["layer4"] = load_json_if_exists(
            store.path("orchestration/layer4_report.json")
        )

    if include_layer5:
        reports["layer5"] = load_json_if_exists(
            store.path("orchestration/layer5_report.json")
        )

    if include_layer6:
        reports["layer6"] = load_json_if_exists(
            store.path("orchestration/layer6_report.json")
        )

    return reports


def stage5_artifact_paths(store: ArtifactStore) -> dict:
    """
    Canonical Stage-5 artifact locations.

    Provider inference currently remains outside the FastAPI process because
    the validated Provider-A and Provider-B environments are intentionally
    isolated.  The API consumes the frozen Stage-5 evidence artifacts.
    """
    return {
        "provider_a_manifest": store.path(
            "L5_provider_A",
            "htr_manifest.json",
        ),
        "provider_b_manifest": store.path(
            "L5_provider_B",
            "htr_manifest.json",
        ),
        "provider_comparison": store.path(
            "L5_compare",
            "provider_comparison.json",
        ),
        "htr_readiness": store.path(
            "L5_readiness",
            "htr_readiness.json",
        ),
    }


def load_stage5_artifact_bundle(store: ArtifactStore) -> dict:
    paths = stage5_artifact_paths(store)

    return {
        "paths": paths,
        "provider_a_manifest": load_json_if_exists(
            paths["provider_a_manifest"]
        ),
        "provider_b_manifest": load_json_if_exists(
            paths["provider_b_manifest"]
        ),
        "provider_comparison": load_json_if_exists(
            paths["provider_comparison"]
        ),
        "htr_readiness": load_json_if_exists(
            paths["htr_readiness"]
        ),
    }



def stage6_artifact_paths(store: ArtifactStore) -> dict:
    """
    Canonical Stage-6 semantic/reconstruction artifact locations.

    At this integration checkpoint, Stage 6A-6F execution remains external to
    the FastAPI process. The endpoint consumes the validated frozen artifacts
    and performs Stage-6G adaptive orchestration in the main process.
    """
    return {
        "stage6a_input_table": store.path(
            "L6",
            "htr_input_table.json",
        ),
        "stage6b_manifest": store.path(
            "L6",
            "stage6b_manifest.json",
        ),
        "stage6c_manifest": store.path(
            "L6",
            "stage6c_manifest.json",
        ),
        "stage6d_manifest": store.path(
            "L6",
            "stage6d_manifest.json",
        ),
        "stage6d2_manifest": store.path(
            "L6",
            "stage6d2_manifest.json",
        ),
        "stage6e_manifest": store.path(
            "L6",
            "stage6e_manifest.json",
        ),
        "stage6f_manifest": store.path(
            "L6",
            "stage6f_manifest.json",
        ),
        "reconstruction_report": store.path(
            "L6",
            "reconstruction_report.json",
        ),
        "transcription_trust": store.path(
            "L6",
            "transcription_trust.json",
        ),
        "retry_state": store.path(
            "orchestration",
            "stage6_retry_state.json",
        ),
    }


def load_stage6_artifact_bundle(store: ArtifactStore) -> dict:
    paths = stage6_artifact_paths(store)

    return {
        "paths": paths,
        "stage6a_input_table": load_json_if_exists(
            paths["stage6a_input_table"]
        ),
        "stage6b_manifest": load_json_if_exists(
            paths["stage6b_manifest"]
        ),
        "stage6c_manifest": load_json_if_exists(
            paths["stage6c_manifest"]
        ),
        "stage6d_manifest": load_json_if_exists(
            paths["stage6d_manifest"]
        ),
        "stage6d2_manifest": load_json_if_exists(
            paths["stage6d2_manifest"]
        ),
        "stage6e_manifest": load_json_if_exists(
            paths["stage6e_manifest"]
        ),
        "stage6f_manifest": load_json_if_exists(
            paths["stage6f_manifest"]
        ),
        "reconstruction_report": load_json_if_exists(
            paths["reconstruction_report"]
        ),
        "transcription_trust": load_json_if_exists(
            paths["transcription_trust"]
        ),
        "retry_state": load_json_if_exists(
            paths["retry_state"]
        ),
    }


def simplify_devanagari_for_display(text: str) -> str:
    """
    Readability-only UI representation.

    Exact provider output is preserved unchanged in artifacts. This display
    view hides characters that commonly render as tofu/square boxes:
      - Vedic Extensions U+1CD0..U+1CFF
      - Devanagari Extended signs U+A8E0..U+A8FF
      - residual combining diacritics U+0300..U+036F
      - variation selectors
      - replacement/white-square placeholders
      - invisible Unicode control/format characters

    This is display normalization only, not linguistic correction.
    """
    text = unicodedata.normalize("NFC", text or "")
    kept: list[str] = []

    for ch in text:
        cp = ord(ch)

        if ch in {"\n", "\t"}:
            kept.append(ch)
            continue

        if (
            0x1CD0 <= cp <= 0x1CFF
            or 0xA8E0 <= cp <= 0xA8FF
            or 0x0300 <= cp <= 0x036F
            or 0xFE00 <= cp <= 0xFE0F
            or cp in {0x25A1, 0x25A0, 0xFFFD}
        ):
            continue

        if unicodedata.category(ch).startswith("C"):
            continue

        kept.append(ch)

    normalized = "".join(kept)

    return "\n".join(
        " ".join(line.split())
        for line in normalized.splitlines()
    ).strip()


def stage5_transcription_view(bundle: dict) -> dict:
    """
    Build a UI-safe view of Provider-A and Provider-B Devanagari output.

    No provider is silently selected as the final scholarly transcription.
    When H(p) is low, both outputs remain explicitly untrusted alternatives.
    """
    comparison = bundle.get("provider_comparison", {}) or {}
    comparison_lines = comparison.get("lines", []) or []

    lines = []

    for row in comparison_lines:
        if not isinstance(row, dict):
            continue

        lines.append(
            {
                "line_id": row.get("line_id"),
                "reading_order": row.get("reading_order"),
                "provider_a_devanagari": row.get(
                    "provider_a_devanagari",
                    "",
                ),
                "provider_b_devanagari": row.get(
                    "provider_b_devanagari",
                    "",
                ),
                "provider_a_devanagari_display": (
                    simplify_devanagari_for_display(
                        row.get("provider_a_devanagari", "")
                    )
                ),
                "provider_b_devanagari_display": (
                    simplify_devanagari_for_display(
                        row.get("provider_b_devanagari", "")
                    )
                ),
                "strict_char_similarity": row.get(
                    "strict_char_similarity"
                ),
                "content_char_similarity": row.get(
                    "content_char_similarity"
                ),
                "length_ratio": row.get("length_ratio"),
                "notes": row.get("notes", []),
            }
        )

    lines.sort(
        key=lambda row: (
            row.get("reading_order")
            if row.get("reading_order") is not None
            else 10**9,
            str(row.get("line_id") or ""),
        )
    )

    provider_a_page = "\n".join(
        row["provider_a_devanagari"]
        for row in lines
        if row["provider_a_devanagari"]
    )

    provider_b_page = "\n".join(
        row["provider_b_devanagari"]
        for row in lines
        if row["provider_b_devanagari"]
    )

    provider_a_page_display = "\n".join(
        row["provider_a_devanagari_display"]
        for row in lines
        if row["provider_a_devanagari_display"]
    )

    provider_b_page_display = "\n".join(
        row["provider_b_devanagari_display"]
        for row in lines
        if row["provider_b_devanagari_display"]
    )

    readiness = bundle.get("htr_readiness", {}) or {}
    readiness_page = readiness.get("page", {}) or {}
    H = readiness_page.get("htr_readiness_H_page")

    return {
        "status": (
            "untrusted_multi_provider_output"
            if lines
            else "unavailable"
        ),
        "H": H,
        "provider_a_page_devanagari": provider_a_page,
        "provider_b_page_devanagari": provider_b_page,
        "provider_a_page_devanagari_display": provider_a_page_display,
        "provider_b_page_devanagari_display": provider_b_page_display,
        "display_normalization": {
            "enabled": True,
            "purpose": (
                "Readability-only browser view; exact provider output remains "
                "preserved in the *_devanagari fields and artifacts."
            ),
            "linguistic_correction_applied": False,
            "vedic_extension_marks_hidden": True,
            "residual_combining_marks_hidden": True,
        },
        "trusted_combined_transcription": None,
        "trusted_combined_transcription_available": False,
        "lines": lines,
        "interpretation": (
            "Provider A and Provider B Devanagari outputs are shown as "
            "alternative HTR hypotheses. They are not a scholar-verified "
            "transcription. No combined/final text is selected while H(p) "
            "and cross-provider agreement remain below the acceptance gates."
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline": "six-stage-research-pipeline",
        "implemented_research_stages": [1, 2, 3, 4, 5, 6],
        "stage5": {
            "orchestration_integrated": True,
            "backend_execution_mode": "integrated_isolated_subprocesses",
            "provider_inference_in_fastapi_process": False,
            "provider_inference_launched_by_fastapi": True,
            "provider_a_environment": "venv-stage5",
            "provider_b_environment": "venv-provider-b",
            "architecture": (
                "FastAPI coordinates isolated HTR provider subprocesses, "
                "then performs provider comparison, H(p) readiness fusion "
                "and adaptive Stage-5 routing."
            ),
        },
        "stage6": {
            "orchestration_integrated": True,
            "backend_execution_mode": "integrated_isolated_subprocesses",
            "semantic_substage_execution_in_fastapi_process": False,
            "provider_inference_launched_by_fastapi": True,
            "stage6_environment": "venv-stage6",
            "stage6_rag_environment": "venv-stage6-rag",
            "signal": "T(p)",
            "architecture": (
                "FastAPI launches Stage 6A-6F in isolated native-ARM "
                "subprocess environments, then executes Stage 6G adaptive "
                "retry / scholar-last routing in the main application process."
            ),
        },
        "planned_research_stages": {
            "translation": (
                "translation after trusted or scholar-validated transcription"
            ),
        },
    }


# ---------------------------------------------------------------------------
# PRE-PROCESSING / STAGE 0
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    execution_profile: str = Form("generic_default"),
):
    os.makedirs("data/raw", exist_ok=True)

    file_id = str(uuid.uuid4())
    input_path = f"data/raw/{file_id}_{file.filename}"

    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    try:
        execution_profile = normalize_execution_profile(execution_profile)
        profile_config = load_execution_profile_config(execution_profile)
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid execution profile.",
                "detail": str(exc),
            },
        )

    run_id = ArtifactStore.new_run_id("web")
    store = ArtifactStore("artifacts", run_id)

    run_layer0_ingest(input_path, store, notes="web upload")
    save_stage_state(
        store,
        "upload",
        {
            "input_path": input_path,
            "execution_profile": execution_profile,
        },
    )
    store.write_json(
        "orchestration/execution_profile.json",
        {
            "profile_id": execution_profile,
            "profile_source": (
                str(PLATFORM_EXECUTION_PROFILE_CONFIGS[execution_profile])
                if PLATFORM_EXECUTION_PROFILE_CONFIGS[execution_profile]
                else "layer_defaults"
            ),
            "stage4_profile_parameters_available": bool(
                profile_config.get("stage4", {}).get(
                    "accepted_showcase_parameters"
                )
            ),
        },
    )

    return {
        "run_id": run_id,
        "pipeline_position": "preprocessing_upload",
        "execution_profile": execution_profile,
        "raw_image": image_to_base64(store.path("L0", "raw.png")),
    }


@app.post("/analyze/{run_id}")
@app.post("/stage1/analyze/{run_id}", include_in_schema=False)
def analyze_condition(run_id: str):
    """
    Pre-processing condition analysis.

    This is deliberately not counted as one of the six research stages.
    The legacy /stage1/analyze route remains as a compatibility alias.
    """
    store = ArtifactStore("artifacts", run_id)
    state = load_stage_state(store, "upload")

    input_path = state.get("input_path")
    if not input_path:
        return JSONResponse(
            status_code=400,
            content={"error": "No uploaded image found for this run_id"},
        )

    img = read_image_bgr(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    diagnostics = {
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
        "uneven_illumination": "Possible" if gray.std() < 55 else "Moderate",
        "faded_ink": "Possible" if gray.mean() > 135 else "Low",
        "background_yellowing": (
            "Possible"
            if img[:, :, 2].mean() > img[:, :, 0].mean() + 10
            else "Low"
        ),
    }

    summary = []
    if diagnostics["faded_ink"] != "Low":
        summary.append("Ink strokes appear faded.")
    if diagnostics["uneven_illumination"] != "Moderate":
        summary.append("Page lighting appears uneven.")
    if diagnostics["background_yellowing"] != "Low":
        summary.append("Background discoloration detected.")
    if not summary:
        summary.append("The manuscript appears visually stable.")

    save_stage_state(store, "preprocessing_analysis", diagnostics)

    return {
        "run_id": run_id,
        "pipeline_position": "preprocessing_analysis",
        "diagnostics": diagnostics,
        "summary": summary,
        "raw_image": image_to_base64(store.path("L0", "raw.png")),
    }


# ---------------------------------------------------------------------------
# SIX-STAGE RESEARCH PIPELINE
# ---------------------------------------------------------------------------

@app.post("/pipeline/stage1/restore/{run_id}")
@app.post("/stage2/restore/{run_id}", include_in_schema=False)
def research_stage1_restore(run_id: str):
    """Research Stage 1 — degradation-aware restoration / readability."""
    store = ArtifactStore("artifacts", run_id)
    state = load_stage_state(store, "upload")

    input_path = state.get("input_path")
    if not input_path:
        return JSONResponse(
            status_code=400,
            content={"error": "No uploaded image found for this run_id"},
        )

    img = read_image_bgr(input_path)

    l1 = run_layer1_restore(
        img,
        store,
        bg_ksize=35,
        clahe_clip=2.0,
        clahe_tile=16,
        gamma=1.0,
        bin_method="otsu",
        preserve_separators=True,
    )

    report1 = evaluate_and_save_layer1(store, l1.metrics)
    save_stage_state(
        store,
        "research_stage1_restore",
        {"status": "done", "layer": "L1"},
    )

    return {
        "run_id": run_id,
        "research_stage": 1,
        "stage_name": "restoration",
        "images": {
            "tone": image_to_base64(store.path("L1", "tone.png")),
            "balanced": image_to_base64(store.path("L1", "balanced.png")),
            "binary": image_to_base64(store.path("L1", "binary.png")),
            "separator_mask": image_to_base64(
                store.path("L1", "separator_mask.png")
            ),
        },
        "orchestration_report": report1,
    }


@app.post("/pipeline/stage2/damage/{run_id}")
@app.post("/stage3/damage/{run_id}", include_in_schema=False)
def research_stage2_damage(run_id: str):
    """Research Stage 2 — damage/interference and uncertainty analysis."""
    store = ArtifactStore("artifacts", run_id)

    balanced_path = store.path("L1", "balanced.png")
    if not os.path.exists(balanced_path):
        return JSONResponse(
            status_code=400,
            content={"error": "Research Stage 1 must be completed first"},
        )

    balanced = cv2.imread(balanced_path, cv2.IMREAD_COLOR)
    if balanced is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Could not read Research Stage 1 balanced image"},
        )

    gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)

    l2 = run_layer2_damage(gray, store)
    report2 = evaluate_and_save_layer2(store, l2.metrics)
    save_stage_state(
        store,
        "research_stage2_damage",
        {"status": "done", "layer": "L2"},
    )

    return {
        "run_id": run_id,
        "research_stage": 2,
        "stage_name": "damage_and_uncertainty",
        "images": {
            "damage_mask": image_to_base64(store.path("L2", "damage_mask.png")),
            "uncertainty_map": image_to_base64(
                store.path("L2", "uncertainty_map.png")
            ),
        },
        "orchestration_report": report2,
    }


@app.post("/pipeline/stage3/layout/{run_id}")
@app.post("/stage4/layout/{run_id}", include_in_schema=False)
def research_stage3_layout(run_id: str):
    """
    Research Stage 3 — layout and coarse text-region analysis.

    This stage identifies text-bearing regions.  It does not claim to have
    segmented individual manuscript lines; that is the responsibility of
    Research Stage 4.
    """
    store = ArtifactStore("artifacts", run_id)

    balanced_path = store.path("L1", "balanced.png")
    binary_path = store.path("L1", "binary.png")

    if not os.path.exists(balanced_path) or not os.path.exists(binary_path):
        return JSONResponse(
            status_code=400,
            content={"error": "Research Stage 1 must be completed first"},
        )

    balanced = cv2.imread(balanced_path, cv2.IMREAD_COLOR)
    binary = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)

    if balanced is None or binary is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Could not read Research Stage 1 artifacts"},
        )

    gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)

    l3 = run_layer3_layout(gray, binary, store)
    report3 = evaluate_and_save_layer3(store, l3.metrics)
    save_stage_state(
        store,
        "research_stage3_layout",
        {"status": "done", "regions": len(l3.regions), "layer": "L3"},
    )

    pre_stage4 = finalize_orchestration(
        store,
        current_layer_reports(store, include_layer4=False),
    )

    return {
        "run_id": run_id,
        "research_stage": 3,
        "stage_name": "layout_and_text_regions",
        "num_regions": len(l3.regions),
        "images": {
            "text_region_mask": image_to_base64(
                store.path("L3", "text_region_mask.png")
            ),
            "layout_overlay": image_to_base64(
                store.path("L3", "layout_overlay.png")
            ),
        },
        "orchestration_report": report3,
        "pre_stage4_orchestration": pre_stage4,
    }


@app.post("/pipeline/stage4/segment/{run_id}")
@app.post("/stage5/segment/{run_id}", include_in_schema=False)
def research_stage4_segment(run_id: str):
    """Research Stage 4 — script-aware, loss-aware line segmentation."""
    store = ArtifactStore("artifacts", run_id)
    upload_state = load_stage_state(store, "upload")

    try:
        execution_profile = normalize_execution_profile(
            upload_state.get("execution_profile", "generic_default")
        )
        stage4_profile_params = stage4_parameters_for_execution_profile(
            execution_profile
        )
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Stage 4 execution profile could not be loaded.",
                "detail": str(exc),
            },
        )

    balanced_path = store.path("L1", "balanced.png")
    binary_path = store.path("L1", "binary.png")
    layout_mask_path = store.path("L3", "text_region_mask.png")
    upstream_uncertainty_path = store.path("L2", "uncertainty_map.png")

    missing = []
    if not os.path.exists(balanced_path):
        missing.append("L1/balanced.png")
    if not os.path.exists(binary_path):
        missing.append("L1/binary.png")
    if not os.path.exists(layout_mask_path):
        missing.append("L3/text_region_mask.png")

    if missing:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Research Stage 4 cannot run because upstream artifacts are missing.",
                "missing_artifacts": missing,
                "required_action": (
                    "Complete Research Stages 1-3 before line segmentation."
                ),
            },
        )

    balanced_gray = cv2.imread(balanced_path, cv2.IMREAD_GRAYSCALE)
    binary = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    layout_mask = cv2.imread(layout_mask_path, cv2.IMREAD_GRAYSCALE)

    if balanced_gray is None or binary is None or layout_mask is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Could not read one or more Stage 4 input artifacts"},
        )

    upstream_uncertainty = None
    if os.path.exists(upstream_uncertainty_path):
        upstream_uncertainty = cv2.imread(
            upstream_uncertainty_path,
            cv2.IMREAD_GRAYSCALE,
        )

    try:
        l4 = run_layer4_segmentation(
            balanced_gray,
            binary,
            store,
            text_region_mask_u8=layout_mask,
            upstream_uncertainty_u8=upstream_uncertainty,
            **stage4_profile_params,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Research Stage 4 segmentation failed.",
                "detail": str(exc),
            },
        )

    report4 = evaluate_and_save_layer4(store, l4.metrics)
    final = finalize_orchestration(
        store,
        current_layer_reports(store, include_layer4=True),
    )

    save_stage_state(
        store,
        "research_stage4_segment",
        {
            "status": "done",
            "layer": "L4",
            "execution_profile": execution_profile,
            "stage4_parameter_source": (
                "validated_profile"
                if stage4_profile_params
                else "layer4_defaults"
            ),
            "stage4_profile_parameters": stage4_profile_params,
            "algorithm_version": l4.metrics.get("algorithm_version"),
            "num_lines": len(l4.lines),
            "segmentation_confidence": l4.metrics.get(
                "segmentation_confidence"
            ),
            "segmentation_readiness": final.get("signals", {}).get("S"),
            "orchestrator_status": final.get("overall_status"),
            "next_action": final.get("next_action"),
        },
    )

    return {
        "run_id": run_id,
        "research_stage": 4,
        "stage_name": "script_aware_line_segmentation",
        "execution_profile": execution_profile,
        "stage4_parameter_source": (
            "validated_profile"
            if stage4_profile_params
            else "layer4_defaults"
        ),
        "algorithm_version": l4.metrics.get("algorithm_version"),
        "num_lines": len(l4.lines),
        "metrics": l4.metrics,
        "images": {
            "segmentation_overlay": image_to_base64(
                store.path("L4", "segmentation_overlay.png")
            ),
            "segmentation_mask": image_to_base64(
                store.path("L4", "segmentation_mask.png")
            ),
            "uncertainty_map": image_to_base64(
                store.path("L4", "uncertainty_map.png")
            ),
            "row_profile_debug": image_to_base64(
                store.path("L4", "row_profile_debug.png")
            ),
        },
        "orchestration_report": report4,
        "final_orchestration": final,
        "line_manifest": f"artifacts/{run_id}/L4/line_manifest.json",
        "next_action": final.get("next_action"),
    }


@app.post("/pipeline/stage5/htr/{run_id}")
def research_stage5_htr(
    run_id: str,
    third_provider_available: bool = False,
):
    """
    Research Stage 5 — integrated sequence-based HTR.

    FastAPI remains in the main application environment while launching:
      - Provider A with venv-stage5/bin/python
      - Provider B with venv-provider-b/bin/python

    After fresh inference, the coordinator performs A/B comparison and Stage-5F
    H(p) readiness fusion.  The main orchestration evaluator then executes
    Stage-5G adaptive routing and updates the unified final decision.

    H(p) is system HTR readiness, not recognition accuracy.
    """
    store = ArtifactStore("artifacts", run_id)

    layer4_report = load_json_if_exists(
        store.path("orchestration/layer4_report.json")
    )

    if not layer4_report:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Research Stage 4 must be completed first.",
                "required_action": (
                    "Complete Stage 4 segmentation and its orchestration "
                    "evaluation before Stage 5."
                ),
            },
        )

    with STAGE5_ACTIVE_RUNS_LOCK:
        if run_id in STAGE5_ACTIVE_RUNS:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "Research Stage 5 is already running for this run_id.",
                    "run_id": run_id,
                    "required_action": (
                        "Wait for the active Stage-5 execution to complete."
                    ),
                },
            )

        STAGE5_ACTIVE_RUNS.add(run_id)

    try:
        runtime_summary = run_stage5_pipeline(
            project_root=PROJECT_ROOT,
            run_id=run_id,
            artifacts=PROJECT_ROOT / "artifacts",
            device="auto",
        )

        bundle = load_stage5_artifact_bundle(store)
        readiness = bundle["htr_readiness"]

        if not readiness:
            raise RuntimeError(
                "Integrated Stage 5 completed provider execution but "
                "htr_readiness.json was not produced."
            )


        # STAGE5_AUTO_PROVIDER_C_V1
        # Provider C is invoked automatically only when segmentation is healthy
        # and baseline A/B HTR readiness is weak. H remains a readiness/evidence
        # signal; it is not accuracy, probability, CER, or WER.
        readiness_page_for_c = readiness.get("page", {}) or {}
        readiness_evidence_for_c = (
            readiness_page_for_c.get("evidence", {}) or {}
        )

        S_for_c = readiness_evidence_for_c.get(
            "segmentation_readiness"
        )
        H_for_c = readiness_page_for_c.get(
            "htr_readiness_H_page"
        )

        try:
            S_for_c = float(S_for_c)
        except (TypeError, ValueError):
            S_for_c = None

        try:
            H_for_c = float(H_for_c)
        except (TypeError, ValueError):
            H_for_c = None

        provider_c_adaptive = {
            "runtime_available": True,
            "trigger_policy": {
                "minimum_segmentation_readiness_S": 0.75,
                "maximum_baseline_htr_readiness_H": 0.50,
            },
            "eligible": bool(
                S_for_c is not None
                and H_for_c is not None
                and S_for_c >= 0.75
                and H_for_c < 0.50
            ),
            "attempted": False,
            "status": "not_triggered",
            "baseline": {
                "S": S_for_c,
                "H_ab": H_for_c,
            },
            "scientific_guardrail": (
                "Provider C supplies independent visual HTR evidence. "
                "Changes in H are readiness/evidence changes and do not establish transcription accuracy without verified ground truth."
            ),
        }

        if provider_c_adaptive["eligible"]:
            provider_c_adaptive["attempted"] = True
            provider_c_adaptive["status"] = "running"

            try:
                provider_c_summary = run_stage5_provider_c_retry(
                    project_root=PROJECT_ROOT,
                    run_id=run_id,
                    artifacts=PROJECT_ROOT / "artifacts",
                )

                provider_c_adaptive.update(
                    {
                        "status": "completed",
                        "provider_c_executed": True,
                        "best_candidate_pair": (
                            provider_c_summary.get(
                                "best_candidate_pair"
                            )
                        ),
                        "best_candidate_H": (
                            provider_c_summary.get(
                                "best_candidate_H"
                            )
                        ),
                        "delta_H": provider_c_summary.get(
                            "delta_H_vs_ab"
                        ),
                        "H_improved": provider_c_summary.get(
                            "H_improved"
                        ),
                        "canonical_ab_readiness_preserved": (
                            provider_c_summary.get(
                                "canonical_ab_readiness_preserved",
                                True,
                            )
                        ),
                        "summary_path": provider_c_summary.get(
                            "summary_path"
                        ),
                        "artifacts": provider_c_summary.get(
                            "artifacts",
                            {},
                        ),
                    }
                )

            except Exception as provider_c_exc:
                # PROVIDER_C_FAILED_ATTEMPT_PROVENANCE_V1
                provider_c_error = (
                    f"{type(provider_c_exc).__name__}: "
                    f"{provider_c_exc}"
                )
                provider_c_adaptive.update(
                    {
                        "status": "failed_ab_preserved",
                        "provider_c_executed": False,
                        "error": provider_c_error,
                        "canonical_ab_readiness_preserved": True,
                    }
                )
                store.write_json(
                    "L5_adaptive_C/provider_c_retry_summary.json",
                    {
                        "schema_version": "1.0-provider-c-failed-attempt",
                        "provider_c_executed": False,
                        "attempted": True,
                        "eligible": True,
                        "runtime_available": True,
                        "status": "failed_ab_preserved",
                        "baseline": {
                            "S": S_for_c,
                            "H_ab": H_for_c,
                        },
                        "error": provider_c_error,
                        "canonical_ab_readiness_preserved": True,
                        "scientific_guardrail": (
                            "Provider C failed after an actual adaptive attempt. "
                            "Canonical H(A+B) remains unchanged; failure is not "
                            "transcription accuracy or Ground Truth."
                        ),
                    },
                )

        provider_c_remaining_available = not bool(
            provider_c_adaptive["attempted"]
        )
        report5 = evaluate_and_save_layer5(
            store,
            readiness,
            third_provider_available=provider_c_remaining_available,
        )

        reports = current_layer_reports(
            store,
            include_layer4=True,
            include_layer5=True,
        )

        final = finalize_orchestration(
            store,
            reports,
        )

        page = readiness.get("page", {})
        readiness_evidence = page.get("evidence", {})

        provider_comparison = bundle.get(
            "provider_comparison",
            {},
        )
        comparison_aggregate = provider_comparison.get(
            "aggregate",
            {},
        )
        transcription = stage5_transcription_view(bundle)

        save_stage_state(
            store,
            "research_stage5_htr",
            {
                "status": "done",
                "layer": "L5",
                "backend_execution_mode": (
                    "integrated_isolated_subprocesses"
                ),
                "provider_inference_executed": True,
                "htr_readiness_H": final.get(
                    "signals",
                    {},
                ).get("H"),
                "segmentation_readiness_S": final.get(
                    "signals",
                    {},
                ).get("S"),
                "orchestrator_status": final.get("overall_status"),
                "htr_decision": final.get("htr_decision"),
                "next_action": final.get("next_action"),
                "third_provider_available": third_provider_available,
                "runtime_seconds": runtime_summary.get(
                    "runtime_seconds"
                ),
            },
        )

        return {
            "run_id": run_id,
            "research_stage": 5,
            "stage_name": "sequence_based_htr",
            "backend_execution_mode": (
                "integrated_isolated_subprocesses"
            ),
            "provider_inference_executed_by_endpoint": True,
            "runtime": {
                "coordinator_version": runtime_summary.get(
                    "coordinator_version"
                ),
                "runtime_seconds": runtime_summary.get(
                    "runtime_seconds"
                ),
                "provider_a": runtime_summary.get("provider_a", {}),
                "provider_b": runtime_summary.get("provider_b", {}),
            },
            "signals": {
                "S": final.get("signals", {}).get("S"),
                "H": final.get("signals", {}).get("H"),
            },
            "adaptive_provider_c": provider_c_adaptive,
            "htr_readiness": {
                "completion": readiness_evidence.get("completion"),
                "script_integrity": readiness_evidence.get(
                    "script_integrity"
                ),
                "decoder_reliability": readiness_evidence.get(
                    "decoder_reliability"
                ),
                "sequence_quality": readiness_evidence.get(
                    "sequence_quality"
                ),
                "cross_provider_agreement": readiness_evidence.get(
                    "cross_provider_agreement"
                ),
                "total_lines": page.get("total_lines"),
                "low_readiness_lines": page.get(
                    "low_readiness_lines"
                ),
                "low_agreement_lines": page.get(
                    "low_agreement_lines"
                ),
                "ground_truth_available": page.get(
                    "ground_truth_available",
                    False,
                ),
                "cer": page.get("cer"),
                "wer": page.get("wer"),
            },
            "transcription": transcription,
            "provider_comparison": {
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
                "low_agreement_lines": comparison_aggregate.get(
                    "low_agreement_lines"
                ),
                "moderate_agreement_lines": comparison_aggregate.get(
                    "moderate_agreement_lines"
                ),
                "high_agreement_lines": comparison_aggregate.get(
                    "high_agreement_lines"
                ),
                "interpretation": (
                    "Cross-provider agreement is consistency evidence only, "
                    "not recognition accuracy."
                ),
            },
            "orchestration_report": report5,
            "final_orchestration": final,
            "next_action": final.get("next_action"),
            "artifacts": {
                **runtime_summary.get("artifacts", {}),
                "layer5_report": (
                    f"artifacts/{run_id}/orchestration/layer5_report.json"
                ),
                "post_stage5_decision": (
                    f"artifacts/{run_id}/orchestration/"
                    "post_stage5_decision.json"
                ),
                "final_decision": (
                    f"artifacts/{run_id}/orchestration/final_decision.json"
                ),
            },
        }

    except Stage5PipelineError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Research Stage 5 integrated execution failed.",
                "failed_stage": exc.stage,
                "detail": str(exc),
                "log_path": exc.log_path,
            },
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Research Stage 5 orchestration failed.",
                "detail": str(exc),
            },
        )

    finally:
        with STAGE5_ACTIVE_RUNS_LOCK:
            STAGE5_ACTIVE_RUNS.discard(run_id)




# GENERIC_PROVIDER_C_ENDPOINT_V1
@app.post("/pipeline/stage5/provider-c/{run_id}")
def research_stage5_provider_c_retry(run_id: str):
    """
    Execute Qwen3-VL / Bedrock Mantle as an independent adaptive HTR retry.

    The original A/B readiness artifact remains unchanged.
    """
    store = ArtifactStore("artifacts", run_id)

    layer4_report = load_json_if_exists(
        store.path("orchestration/layer4_report.json")
    )
    stage5_bundle = load_stage5_artifact_bundle(store)
    readiness_ab = stage5_bundle.get("htr_readiness", {}) or {}

    if not layer4_report or not readiness_ab:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Provider C requires completed Stage 4 and baseline "
                    "Stage-5 A/B execution."
                ),
                "required_action": "Complete Stage 5 A/B before Provider C.",
            },
        )

    with STAGE5_ACTIVE_RUNS_LOCK:
        if run_id in STAGE5_ACTIVE_RUNS:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "Stage 5 is already running for this run_id.",
                    "run_id": run_id,
                },
            )
        STAGE5_ACTIVE_RUNS.add(run_id)

    try:
        summary = run_stage5_provider_c_retry(
            project_root=PROJECT_ROOT,
            run_id=run_id,
            artifacts=PROJECT_ROOT / "artifacts",
        )

        manifest_c = load_json_if_exists(
            store.path("L5_provider_C/htr_manifest.json")
        )

        lines = []
        for row in manifest_c.get("lines", []) if manifest_c else []:
            lines.append(
                {
                    "line_id": row.get("line_id"),
                    "reading_order": row.get("reading_order"),
                    "devanagari_text": row.get("devanagari_text")
                    or row.get("raw_text")
                    or "",
                    "review_required": row.get("review_required"),
                    "runtime_ms": row.get("runtime_ms"),
                }
            )

        return {
            "run_id": run_id,
            "research_stage": 5,
            "stage_name": "adaptive_visual_htr_provider_c",
            "provider_c": {
                "provider_id": (
                    manifest_c.get("provider", {}).get("provider_id")
                    if manifest_c
                    else "qwen_bedrock_mantle"
                ),
                "model_id": (
                    manifest_c.get("provider", {}).get("model_id")
                    if manifest_c
                    else None
                ),
                "recognized_lines": (
                    manifest_c.get("metrics", {}).get("recognized_lines")
                    if manifest_c
                    else None
                ),
                "lines": lines,
            },
            "adaptive_evidence": {
                "baseline_pair": summary.get("baseline", {}).get("pair"),
                "H_ab": summary.get("baseline", {}).get("H"),
                "H_ac": summary.get("candidate_pairs", {}).get("A+C", {}).get("H"),
                "H_bc": summary.get("candidate_pairs", {}).get("B+C", {}).get("H"),
                "best_candidate_pair": summary.get("best_candidate_pair"),
                "best_candidate_H": summary.get("best_candidate_H"),
                "delta_H": summary.get("delta_H_vs_ab"),
                "H_improved": summary.get("H_improved"),
            },
            "scientific_interpretation": summary.get(
                "scientific_interpretation"
            ),
            "canonical_ab_readiness_preserved": True,
            "runtime_seconds": summary.get("runtime_seconds"),
            "artifacts": summary.get("artifacts", {}),
        }

    except Stage5PipelineError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Provider-C adaptive HTR execution failed.",
                "failed_stage": exc.stage,
                "detail": str(exc),
                "log_path": exc.log_path,
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Provider-C adaptive HTR execution failed.",
                "detail": str(exc),
            },
        )
    finally:
        with STAGE5_ACTIVE_RUNS_LOCK:
            STAGE5_ACTIVE_RUNS.discard(run_id)


@app.post("/pipeline/stage6/run/{run_id}")
def research_stage6_run(
    run_id: str,
):
    """
    Research Stage 6 — fully integrated isolated execution.

    FastAPI stays in the original application environment while the Stage-6
    coordinator launches:
      - 6A, 6B, 6C, 6D, 6E, 6F with venv-stage6/bin/python
      - 6D.2 with venv-stage6-rag/bin/python

    After T(p) is produced, the main FastAPI process executes Stage 6G
    scholar-last adaptive routing.

    The endpoint advertises only capabilities that are executable today.
    Provider C / additional machine retry capabilities remain False until
    corresponding runtime implementations actually exist.
    """
    store = ArtifactStore("artifacts", run_id)
    upload_state = load_stage_state(store, "upload")

    try:
        execution_profile = normalize_execution_profile(
            upload_state.get("execution_profile", "generic_default")
        )
        execution_profile_config = load_execution_profile_config(
            execution_profile
        )
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": (
                    "Stage 6 execution profile could not be loaded."
                ),
                "detail": str(exc),
            },
        )

    layer4_report = load_json_if_exists(
        store.path("orchestration/layer4_report.json")
    )

    stage5_bundle = load_stage5_artifact_bundle(store)
    stage5_readiness = stage5_bundle.get("htr_readiness", {}) or {}

    missing = []

    if not layer4_report:
        missing.append("orchestration/layer4_report.json")

    if not stage5_readiness:
        missing.append("L5_readiness/htr_readiness.json")

    if missing:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Research Stage 6 cannot run because Stage 4/5 "
                    "prerequisites are missing."
                ),
                "missing_artifacts": missing,
                "required_action": (
                    "Complete Research Stage 5 before integrated Stage 6."
                ),
            },
        )

    with STAGE6_ACTIVE_RUNS_LOCK:
        if run_id in STAGE6_ACTIVE_RUNS:
            return JSONResponse(
                status_code=409,
                content={
                    "error": (
                        "Research Stage 6 is already running for this run_id."
                    ),
                    "run_id": run_id,
                    "required_action": (
                        "Wait for the active Stage-6 execution to complete."
                    ),
                },
            )

        STAGE6_ACTIVE_RUNS.add(run_id)

    try:
        runtime_summary = run_stage6_pipeline(
            project_root=PROJECT_ROOT,
            run_id=run_id,
            artifacts=PROJECT_ROOT / "artifacts",
            vidyut_data=PROJECT_ROOT / "models" / "vidyut-0.4.0",
            corpus=(
                PROJECT_ROOT
                / "knowledge"
                / "stage6d_dcs"
                / "passages.jsonl"
            ),
        )

        stage6_bundle = load_stage6_artifact_bundle(store)

        stage6e = stage6_bundle.get("stage6e_manifest", {}) or {}
        stage6f = stage6_bundle.get("stage6f_manifest", {}) or {}
        retry_state = stage6_bundle.get("retry_state", {}) or {}

        # STAGE6G_PROVIDER_C_RETRY_PROVENANCE_V2
        # Provider C already executed during Stage 5 is an attempted machine
        # strategy, not an unused future capability. Provenance only:
        # H(A+B), T(p), CER/WER, thresholds, and reconstruction are unchanged.
        provider_c_retry_summary = load_json_if_exists(
            store.path(
                "L5_adaptive_C",
                "provider_c_retry_summary.json",
            )
        )
        provider_c_manifest = load_json_if_exists(
            store.path(
                "L5_provider_C",
                "htr_manifest.json",
            )
        )

        provider_c_already_attempted = bool(
            provider_c_retry_summary
            and provider_c_retry_summary.get(
                "provider_c_executed",
                False,
            )
            and provider_c_manifest
        )

        if provider_c_already_attempted:
            retry_state = dict(retry_state)
            attempted_actions = [
                str(value)
                for value in retry_state.get(
                    "attempted_actions",
                    [],
                )
                if value
            ]

            if "run_third_htr_provider" not in attempted_actions:
                attempted_actions.append(
                    "run_third_htr_provider"
                )

            try:
                existing_attempt_count = int(
                    retry_state.get("attempt_count", 0) or 0
                )
            except (TypeError, ValueError):
                existing_attempt_count = 0

            retry_state["attempted_actions"] = attempted_actions
            retry_state["attempt_count"] = max(
                existing_attempt_count,
                len(set(attempted_actions)),
                1,
            )

            provenance = dict(
                retry_state.get("provenance", {}) or {}
            )
            provenance["provider_c_stage5_adaptive_attempt"] = {
                "recorded": True,
                "action": "run_third_htr_provider",
                "source_summary": (
                    "L5_adaptive_C/provider_c_retry_summary.json"
                ),
                "source_manifest": (
                    "L5_provider_C/htr_manifest.json"
                ),
                "provider_c_executed": True,
                "canonical_ab_readiness_preserved": bool(
                    provider_c_retry_summary.get(
                        "canonical_ab_readiness_preserved",
                        False,
                    )
                ),
                "scientific_guardrail": (
                    "Retry accounting is provenance only; Provider C is "
                    "candidate visual HTR evidence, not ground truth or accuracy."
                ),
            }
            retry_state["provenance"] = provenance

            store.write_json(
                "orchestration/stage6_retry_state.json",
                retry_state,
            )

        if not stage6e:
            raise RuntimeError(
                "Integrated Stage 6 completed but stage6e_manifest.json "
                "was not produced."
            )

        if not stage6f:
            raise RuntimeError(
                "Integrated Stage 6 completed but stage6f_manifest.json "
                "was not produced."
            )

        # Refresh Stage-5 routing under the scholar-last policy.
        report5 = evaluate_and_save_layer5(
            store,
            stage5_readiness,
            third_provider_available=False,
        )

        # Current executable post-Stage-6 retry capabilities are deliberately
        # False. They must become True only when corresponding runtimes exist.
        report6 = evaluate_and_save_layer6(
            store,
            stage6f,
            stage6e,
            stage5_readiness,
            layer4_report,
            third_provider_available=False,
            alternate_visual_htr_available=False,
            expanded_context_retry_available=False,
            retry_state=retry_state,
        )

        reports = current_layer_reports(
            store,
            include_layer4=True,
            include_layer5=True,
            include_layer6=True,
        )

        reports["layer5"] = report5
        reports["layer6"] = report6

        final = finalize_orchestration(
            store,
            reports,
        )

        stage6_cfg = (
            execution_profile_config.get("stage6", {}) or {}
        )
        candidate_mode = bool(
            stage6_cfg.get(
                "candidate_output_mode_when_untrusted",
                False,
            )
        )
        final_requires_review = (
            final.get("overall_status") == "review_required"
        )

        semantic_assist = {
            "enabled": bool(
                stage6_cfg.get(
                    "qwen_semantic_assist_enabled",
                    False,
                )
            ),
            "executed": False,
            "status": "not_required",
            "changes_S_H_T": False,
        }

        if candidate_mode and final_requires_review:
            semantic_assist = (
                run_stage6_semantic_assist_subprocess(
                    run_id=run_id,
                    execution_profile=execution_profile,
                    profile_config=execution_profile_config,
                )
            )

        save_stage_state(
            store,
            "research_stage6_semantic",
            {
                "status": "done",
                "layer": "L6",
                "execution_mode": "integrated_isolated_subprocesses",
                "execution_profile": execution_profile,
                "semantic_assist": semantic_assist,
                "coordinator_version": runtime_summary.get(
                    "coordinator_version"
                ),
                "runtime_seconds": runtime_summary.get(
                    "runtime_seconds"
                ),
                "semantic_transcription_trust_T": final.get(
                    "signals",
                    {},
                ).get("T"),
                "htr_readiness_H": final.get(
                    "signals",
                    {},
                ).get("H"),
                "segmentation_readiness_S": final.get(
                    "signals",
                    {},
                ).get("S"),
                "orchestrator_status": final.get(
                    "overall_status"
                ),
                "semantic_decision": final.get(
                    "semantic_decision"
                ),
                "failure_domain": final.get(
                    "failure_domain"
                ),
                "next_action": final.get(
                    "next_action"
                ),
                "machine_retry_exhausted": final.get(
                    "machine_retry_exhausted",
                    False,
                ),
                "scholar_review_required": final.get(
                    "scholar_review_required",
                    False,
                ),
            },
        )

        return {
            "run_id": run_id,
            "research_stage": 6,
            "stage_name": "semantic_interpretation_and_trust",
            "execution_profile": execution_profile,
            "semantic_assist": semantic_assist,
            "backend_execution_mode": "integrated_isolated_subprocesses",
            "runtime": {
                "coordinator_version": runtime_summary.get(
                    "coordinator_version"
                ),
                "runtime_seconds": runtime_summary.get(
                    "runtime_seconds"
                ),
                "substage_plan": runtime_summary.get(
                    "substage_plan"
                ),
                "substages": runtime_summary.get(
                    "substages"
                ),
                "environments": runtime_summary.get(
                    "environments"
                ),
            },
            "signals": {
                "S": final.get("signals", {}).get("S"),
                "H": final.get("signals", {}).get("H"),
                "T": final.get("signals", {}).get("T"),
            },
            "stage6": {
                "trust_status": stage6f.get(
                    "metrics",
                    {},
                ).get("page_T_status"),
                "unresolved_lines": stage6f.get(
                    "metrics",
                    {},
                ).get("unresolved_lines"),
                "adaptive_retry_required_lines": stage6f.get(
                    "metrics",
                    {},
                ).get("adaptive_retry_required_lines"),
                "translation_eligible_lines": stage6f.get(
                    "metrics",
                    {},
                ).get("translation_eligible_lines"),
                "reconstructed_lines": stage6e.get(
                    "metrics",
                    {},
                ).get("reconstructed_lines"),
                "abstained_lines": stage6e.get(
                    "metrics",
                    {},
                ).get("abstained_lines"),
                "normalized_lines": stage6e.get(
                    "metrics",
                    {},
                ).get("normalized_lines"),
            },
            "adaptive_routing": {
                "routing_version": report6.get(
                    "routing_version"
                ),
                "decision": report6.get(
                    "decision"
                ),
                "next_action": report6.get(
                    "next_action"
                ),
                "failure_domain": report6.get(
                    "failure_domain"
                ),
                "selected_retry_action": report6.get(
                    "selected_retry_action"
                ),
                "machine_retry_exhausted": report6.get(
                    "machine_retry_exhausted"
                ),
                "machine_retry_exhaustion_reason": report6.get(
                    "machine_retry_exhaustion_reason"
                ),
                "scholar_review_required": report6.get(
                    "scholar_review_required"
                ),
                "capabilities": report6.get(
                    "capabilities"
                ),
                "retry_state": report6.get(
                    "retry_state"
                ),
                "recommendations": report6.get(
                    "recommendations"
                ),
            },
            "orchestration_report": report6,
            "final_orchestration": final,
            "next_action": final.get("next_action"),
            "artifacts": {
                "stage6_runtime_summary": (
                    f"artifacts/{run_id}/stage6_runtime/"
                    "stage6_pipeline_summary.json"
                ),
                "layer6_report": (
                    f"artifacts/{run_id}/orchestration/layer6_report.json"
                ),
                "post_stage6_decision": (
                    f"artifacts/{run_id}/orchestration/"
                    "post_stage6_decision.json"
                ),
                "final_decision": (
                    f"artifacts/{run_id}/orchestration/final_decision.json"
                ),
            },
        }

    except Stage6PipelineError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Research Stage 6 integrated execution failed.",
                "failed_stage": exc.stage,
                "detail": str(exc),
                "log_path": exc.log_path,
            },
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Research Stage 6 orchestration failed.",
                "detail": str(exc),
            },
        )

    finally:
        with STAGE6_ACTIVE_RUNS_LOCK:
            STAGE6_ACTIVE_RUNS.discard(run_id)


@app.post("/pipeline/stage6/finalize/{run_id}")
def research_stage6_finalize(
    run_id: str,
    third_provider_available: bool = False,
    alternate_visual_htr_available: bool = False,
    expanded_context_retry_available: bool = False,
):
    """
    Research Stage 6 — semantic/transcription orchestration checkpoint.

    This endpoint intentionally does NOT import Vidyut, RapidFuzz, sentence
    transformers or other Stage-6 specialist dependencies into the FastAPI
    process. It consumes already validated Stage 6E/6F artifacts and runs the
    Stage-6G adaptive controller.

    Scholar review is permitted only by Stage 6G after Stage-6 trust evaluation
    and machine-retry exhaustion under the explicitly declared capabilities.
    """
    store = ArtifactStore("artifacts", run_id)

    layer4_report = load_json_if_exists(
        store.path("orchestration/layer4_report.json")
    )

    stage5_bundle = load_stage5_artifact_bundle(store)
    stage5_readiness = stage5_bundle.get("htr_readiness", {}) or {}

    stage6_bundle = load_stage6_artifact_bundle(store)
    stage6e = stage6_bundle.get("stage6e_manifest", {}) or {}
    stage6f = stage6_bundle.get("stage6f_manifest", {}) or {}
    retry_state = stage6_bundle.get("retry_state", {}) or {}

    # STAGE6G_PROVIDER_C_RETRY_PROVENANCE_V2
    # Provider C already executed during Stage 5 is an attempted machine
    # strategy, not an unused future capability. Provenance only:
    # H(A+B), T(p), CER/WER, thresholds, and reconstruction are unchanged.
    provider_c_retry_summary = load_json_if_exists(
        store.path(
            "L5_adaptive_C",
            "provider_c_retry_summary.json",
        )
    )
    provider_c_manifest = load_json_if_exists(
        store.path(
            "L5_provider_C",
            "htr_manifest.json",
        )
    )

    provider_c_already_attempted = bool(
        provider_c_retry_summary
        and provider_c_retry_summary.get(
            "provider_c_executed",
            False,
        )
        and provider_c_manifest
    )

    if provider_c_already_attempted:
        retry_state = dict(retry_state)
        attempted_actions = [
            str(value)
            for value in retry_state.get(
                "attempted_actions",
                [],
            )
            if value
        ]

        if "run_third_htr_provider" not in attempted_actions:
            attempted_actions.append(
                "run_third_htr_provider"
            )

        try:
            existing_attempt_count = int(
                retry_state.get("attempt_count", 0) or 0
            )
        except (TypeError, ValueError):
            existing_attempt_count = 0

        retry_state["attempted_actions"] = attempted_actions
        retry_state["attempt_count"] = max(
            existing_attempt_count,
            len(set(attempted_actions)),
            1,
        )

        provenance = dict(
            retry_state.get("provenance", {}) or {}
        )
        provenance["provider_c_stage5_adaptive_attempt"] = {
            "recorded": True,
            "action": "run_third_htr_provider",
            "source_summary": (
                "L5_adaptive_C/provider_c_retry_summary.json"
            ),
            "source_manifest": (
                "L5_provider_C/htr_manifest.json"
            ),
            "provider_c_executed": True,
            "canonical_ab_readiness_preserved": bool(
                provider_c_retry_summary.get(
                    "canonical_ab_readiness_preserved",
                    False,
                )
            ),
            "scientific_guardrail": (
                "Retry accounting is provenance only; Provider C is "
                "candidate visual HTR evidence, not ground truth or accuracy."
            ),
        }
        retry_state["provenance"] = provenance

        store.write_json(
            "orchestration/stage6_retry_state.json",
            retry_state,
        )

    missing = []

    if not layer4_report:
        missing.append("orchestration/layer4_report.json")

    if not stage5_readiness:
        missing.append("L5_readiness/htr_readiness.json")

    if not stage6e:
        missing.append("L6/stage6e_manifest.json")

    if not stage6f:
        missing.append("L6/stage6f_manifest.json")

    if missing:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Research Stage 6 orchestration cannot finalize because "
                    "required frozen artifacts are missing."
                ),
                "missing_artifacts": missing,
                "required_action": (
                    "Complete Stage 5 and Stage 6A-6F before calling the "
                    "Stage-6 orchestration checkpoint."
                ),
            },
        )

    try:
        # Refresh Stage-5 routing under the current scholar-last policy. This
        # also repairs older persisted runs whose Layer-5 report predates the
        # Stage-5 v0.2 routing change.
        report5 = evaluate_and_save_layer5(
            store,
            stage5_readiness,
            third_provider_available=third_provider_available,
        )

        report6 = evaluate_and_save_layer6(
            store,
            stage6f,
            stage6e,
            stage5_readiness,
            layer4_report,
            third_provider_available=third_provider_available,
            alternate_visual_htr_available=alternate_visual_htr_available,
            expanded_context_retry_available=expanded_context_retry_available,
            retry_state=retry_state,
        )

        reports = current_layer_reports(
            store,
            include_layer4=True,
            include_layer5=True,
        )

        # Ensure the freshly refreshed reports are used even if a filesystem
        # read races with an external process.
        reports["layer5"] = report5
        reports["layer6"] = report6

        final = finalize_orchestration(
            store,
            reports,
        )

        save_stage_state(
            store,
            "research_stage6_semantic",
            {
                "status": "done",
                "layer": "L6",
                "execution_mode": (
                    "frozen_stage6_artifacts_then_stage6g_routing"
                ),
                "semantic_transcription_trust_T": final.get(
                    "signals",
                    {},
                ).get("T"),
                "htr_readiness_H": final.get(
                    "signals",
                    {},
                ).get("H"),
                "segmentation_readiness_S": final.get(
                    "signals",
                    {},
                ).get("S"),
                "orchestrator_status": final.get(
                    "overall_status"
                ),
                "semantic_decision": final.get(
                    "semantic_decision"
                ),
                "failure_domain": final.get(
                    "failure_domain"
                ),
                "next_action": final.get(
                    "next_action"
                ),
                "machine_retry_exhausted": final.get(
                    "machine_retry_exhausted",
                    False,
                ),
                "scholar_review_required": final.get(
                    "scholar_review_required",
                    False,
                ),
                "capabilities": report6.get(
                    "capabilities",
                    {},
                ),
            },
        )

        return {
            "run_id": run_id,
            "research_stage": 6,
            "stage_name": "semantic_interpretation_and_trust",
            "execution_mode": (
                "frozen_stage6_artifacts_then_stage6g_routing"
            ),
            "signals": {
                "S": final.get("signals", {}).get("S"),
                "H": final.get("signals", {}).get("H"),
                "T": final.get("signals", {}).get("T"),
            },
            "stage6": {
                "trust_status": stage6f.get(
                    "metrics",
                    {},
                ).get("page_T_status"),
                "unresolved_lines": stage6f.get(
                    "metrics",
                    {},
                ).get("unresolved_lines"),
                "adaptive_retry_required_lines": stage6f.get(
                    "metrics",
                    {},
                ).get("adaptive_retry_required_lines"),
                "translation_eligible_lines": stage6f.get(
                    "metrics",
                    {},
                ).get("translation_eligible_lines"),
                "reconstructed_lines": stage6e.get(
                    "metrics",
                    {},
                ).get("reconstructed_lines"),
                "abstained_lines": stage6e.get(
                    "metrics",
                    {},
                ).get("abstained_lines"),
                "normalized_lines": stage6e.get(
                    "metrics",
                    {},
                ).get("normalized_lines"),
            },
            "adaptive_routing": {
                "routing_version": report6.get(
                    "routing_version"
                ),
                "decision": report6.get(
                    "decision"
                ),
                "next_action": report6.get(
                    "next_action"
                ),
                "failure_domain": report6.get(
                    "failure_domain"
                ),
                "selected_retry_action": report6.get(
                    "selected_retry_action"
                ),
                "machine_retry_exhausted": report6.get(
                    "machine_retry_exhausted"
                ),
                "machine_retry_exhaustion_reason": report6.get(
                    "machine_retry_exhaustion_reason"
                ),
                "scholar_review_required": report6.get(
                    "scholar_review_required"
                ),
                "capabilities": report6.get(
                    "capabilities"
                ),
                "retry_state": report6.get(
                    "retry_state"
                ),
                "recommendations": report6.get(
                    "recommendations"
                ),
            },
            "orchestration_report": report6,
            "final_orchestration": final,
            "next_action": final.get(
                "next_action"
            ),
            "artifacts": {
                "layer6_report": (
                    f"artifacts/{run_id}/orchestration/layer6_report.json"
                ),
                "post_stage6_decision": (
                    f"artifacts/{run_id}/orchestration/"
                    "post_stage6_decision.json"
                ),
                "final_decision": (
                    f"artifacts/{run_id}/orchestration/final_decision.json"
                ),
                "stage6e_manifest": (
                    f"artifacts/{run_id}/L6/stage6e_manifest.json"
                ),
                "stage6f_manifest": (
                    f"artifacts/{run_id}/L6/stage6f_manifest.json"
                ),
            },
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Research Stage 6 orchestration failed.",
                "detail": str(exc),
            },
        )



# ---------------------------------------------------------------------------
# ORCHESTRATION REPORTING
# ---------------------------------------------------------------------------

@app.get("/orchestration/{run_id}")
def get_orchestration_report(run_id: str):
    store = ArtifactStore("artifacts", run_id)

    reports = current_layer_reports(
        store,
        include_layer4=True,
        include_layer5=True,
        include_layer6=True,
    )

    final = load_json_if_exists(
        store.path("orchestration/final_decision.json")
    )

    pre_stage4 = load_json_if_exists(
        store.path("orchestration/pre_stage4_decision.json")
    )

    post_stage5 = load_json_if_exists(
        store.path("orchestration/post_stage5_decision.json")
    )

    post_stage6 = load_json_if_exists(
        store.path("orchestration/post_stage6_decision.json")
    )

    stage5_bundle = load_stage5_artifact_bundle(store)
    stage6_bundle = load_stage6_artifact_bundle(store)

    image_paths = {
        "raw": store.path("L0", "raw.png"),
        "balanced": store.path("L1", "balanced.png"),
        "damage_mask": store.path("L2", "damage_mask.png"),
        "layout_overlay": store.path("L3", "layout_overlay.png"),
        "segmentation_overlay": store.path("L4", "segmentation_overlay.png"),
        "segmentation_uncertainty": store.path("L4", "uncertainty_map.png"),
        "row_profile_debug": store.path("L4", "row_profile_debug.png"),
    }

    images = {
        key: image_to_base64(path)
        for key, path in image_paths.items()
        if os.path.exists(path)
    }

    readiness = stage5_bundle.get("htr_readiness", {})
    readiness_page = readiness.get("page", {})

    comparison = stage5_bundle.get("provider_comparison", {})
    comparison_aggregate = comparison.get("aggregate", {})
    transcription = stage5_transcription_view(stage5_bundle)

    stage5_evidence = dict(
        readiness_page.get("evidence", {}) or {}
    )
    stage5_evidence.update(
        {
            "total_lines": readiness_page.get("total_lines"),
            "low_readiness_lines": readiness_page.get(
                "low_readiness_lines"
            ),
            "low_agreement_lines": readiness_page.get(
                "low_agreement_lines"
            ),
        }
    )

    return {
        "run_id": run_id,
        "pipeline": {
            "name": "six-stage-research-pipeline",
            "implemented_stages": [1, 2, 3, 4, 5, 6],
            "stage5_backend_execution_mode": "integrated_isolated_subprocesses",
            "stage6_backend_execution_mode": (
                "integrated_isolated_subprocesses"
            ),
            "planned_stages": {
                "translation": (
                    "translation after trusted or scholar-validated transcription"
                ),
            },
        },
        "layer_reports": reports,
        "pre_stage4_decision": pre_stage4,
        "post_stage5_decision": post_stage5,
        "post_stage6_decision": post_stage6,
        "final_decision": final,
        "stage5": {
            "readiness_version": readiness.get("readiness_version"),
            "H": readiness_page.get("htr_readiness_H_page"),
            "evidence": stage5_evidence,
            "ground_truth_available": readiness_page.get(
                "ground_truth_available",
                False,
            ),
            "cer": readiness_page.get("cer"),
            "wer": readiness_page.get("wer"),
            "transcription": transcription,
            "provider_comparison": {
                "comparison_version": comparison.get(
                    "comparison_version"
                ),
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
                "low_agreement_lines": comparison_aggregate.get(
                    "low_agreement_lines"
                ),
                "moderate_agreement_lines": comparison_aggregate.get(
                    "moderate_agreement_lines"
                ),
                "high_agreement_lines": comparison_aggregate.get(
                    "high_agreement_lines"
                ),
            },
        },
        "stage6": {
            "stage6e_version": stage6_bundle.get(
                "stage6e_manifest",
                {},
            ).get("version"),
            "stage6f_version": stage6_bundle.get(
                "stage6f_manifest",
                {},
            ).get("version"),
            "T": stage6_bundle.get(
                "stage6f_manifest",
                {},
            ).get(
                "metrics",
                {},
            ).get("page_T"),
            "T_status": stage6_bundle.get(
                "stage6f_manifest",
                {},
            ).get(
                "metrics",
                {},
            ).get("page_T_status"),
            "unresolved_lines": stage6_bundle.get(
                "stage6f_manifest",
                {},
            ).get(
                "metrics",
                {},
            ).get("unresolved_lines"),
            "translation_eligible_lines": stage6_bundle.get(
                "stage6f_manifest",
                {},
            ).get(
                "metrics",
                {},
            ).get("translation_eligible_lines"),
            "reconstructed_lines": stage6_bundle.get(
                "stage6e_manifest",
                {},
            ).get(
                "metrics",
                {},
            ).get("reconstructed_lines"),
            "abstained_lines": stage6_bundle.get(
                "stage6e_manifest",
                {},
            ).get(
                "metrics",
                {},
            ).get("abstained_lines"),
            "layer6_report": reports.get(
                "layer6",
                {},
            ),
        },
        "images": images,
    }

# GENERIC_PLATFORM_UI_STATE_V1
@app.get("/pipeline/ui-state/{run_id}")
def generic_platform_ui_state(run_id: str):
    """
    Read-only presentation contract for the generic /platform workspace.

    This endpoint does not execute HTR, reconstruction, translation, retries,
    or scientific scoring. It only projects already-persisted Stage-5/Stage-6
    artifacts into a browser-safe evidence view.
    """
    store = ArtifactStore("artifacts", run_id)

    stage5_bundle = load_stage5_artifact_bundle(store)
    readiness = stage5_bundle.get("htr_readiness", {}) or {}
    readiness_page = readiness.get("page", {}) or {}
    comparison = stage5_bundle.get("provider_comparison", {}) or {}
    comparison_aggregate = comparison.get("aggregate", {}) or {}
    transcription = stage5_transcription_view(stage5_bundle)

    manifest_c = load_json_if_exists(
        store.path("L5_provider_C", "htr_manifest.json")
    )
    provider_c_summary = load_json_if_exists(
        store.path(
            "L5_adaptive_C",
            "provider_c_retry_summary.json",
        )
    )

    provider_c_lines = []
    if isinstance(manifest_c, dict):
        for row in manifest_c.get("lines", []) or []:
            if not isinstance(row, dict):
                continue

            exact = str(
                row.get(
                    "devanagari_text",
                    row.get("devanagari", ""),
                )
                or ""
            )

            provider_c_lines.append(
                {
                    "line_id": row.get("line_id"),
                    "reading_order": row.get("reading_order"),
                    "devanagari_text": exact,
                    "devanagari_display": (
                        simplify_devanagari_for_display(exact)
                    ),
                    "raw_text": row.get("raw_text"),
                    "review_required": row.get("review_required"),
                    "runtime_ms": row.get("runtime_ms"),
                }
            )

    provider_c_page_display = "\n".join(
        str(row.get("devanagari_display", "") or "")
        for row in provider_c_lines
        if row.get("devanagari_display")
    )

    provider_meta = (
        manifest_c.get("provider", {})
        if isinstance(manifest_c, dict)
        else {}
    ) or {}

    baseline = (
        provider_c_summary.get("baseline", {})
        if isinstance(provider_c_summary, dict)
        else {}
    ) or {}
    candidate_pairs = (
        provider_c_summary.get("candidate_pairs", {})
        if isinstance(provider_c_summary, dict)
        else {}
    ) or {}

    ac = candidate_pairs.get("A+C", {}) or {}
    bc = candidate_pairs.get("B+C", {}) or {}

    provider_c_executed = bool(
        (
            provider_c_summary.get("provider_c_executed", False)
            if isinstance(provider_c_summary, dict)
            else False
        )
        or provider_c_lines
    )

    # PROVIDER_C_FAILED_ATTEMPT_UI_STATE_V1
    provider_c_summary_status = (
        provider_c_summary.get("status")
        if isinstance(provider_c_summary, dict)
        else None
    )

    adaptive_evidence = {
        "attempted": bool(provider_c_summary),
        "status": (
            "completed"
            if provider_c_executed
            else (
                provider_c_summary_status
                or ("failed" if provider_c_summary else "not_attempted")
            )
        ),
        "provider_c_executed": provider_c_executed,
        "H_ab": baseline.get(
            "H",
            readiness_page.get("htr_readiness_H_page"),
        ),
        "H_ac": ac.get("H"),
        "H_bc": bc.get("H"),
        "best_candidate_pair": (
            provider_c_summary.get("best_candidate_pair")
            if isinstance(provider_c_summary, dict)
            else None
        ),
        "best_candidate_H": (
            provider_c_summary.get("best_candidate_H")
            if isinstance(provider_c_summary, dict)
            else None
        ),
        "delta_H": (
            provider_c_summary.get("delta_H_vs_ab")
            if isinstance(provider_c_summary, dict)
            else None
        ),
        "H_improved": (
            provider_c_summary.get("H_improved")
            if isinstance(provider_c_summary, dict)
            else None
        ),
        "canonical_ab_readiness_preserved": (
            provider_c_summary.get(
                "canonical_ab_readiness_preserved",
                True,
            )
            if isinstance(provider_c_summary, dict)
            else True
        ),
        "scientific_guardrail": (
            "H(A+B) remains the canonical Stage-5 readiness signal. "
            "H(A+C), H(B+C) and delta-H are comparative adaptive "
            "evidence only; they are not recognition accuracy."
        ),
    }

    stage6_bundle = load_stage6_artifact_bundle(store)
    stage6e = stage6_bundle.get("stage6e_manifest", {}) or {}
    stage6f = stage6_bundle.get("stage6f_manifest", {}) or {}
    stage6e_metrics = stage6e.get("metrics", {}) or {}
    stage6f_metrics = stage6f.get("metrics", {}) or {}

    layer6_report = load_json_if_exists(
        store.path("orchestration", "layer6_report.json")
    )
    final_decision = load_json_if_exists(
        store.path("orchestration", "final_decision.json")
    )

    # GENERIC_PLATFORM_STAGE6_SEMANTIC_ASSIST_UI_V1
    # Read-only projection of already-persisted candidate/audit artifacts.
    # This presentation contract never executes models and never changes S/H/T.
    stage6_semantic_state = load_json_if_exists(
        store.path("state", "research_stage6_semantic.json")
    )
    stage6_semantic_audit = load_json_if_exists(
        store.path(
            "L6_semantic_assist",
            "stage6_semantic_audit.json",
        )
    )
    stage6_semantic_candidate = load_json_if_exists(
        store.path(
            "L6_semantic_assist",
            "stage6_semantic_candidate.json",
        )
    )

    semantic_summary = (
        stage6_semantic_state.get("semantic_assist", {})
        if isinstance(stage6_semantic_state, dict)
        else {}
    ) or {}
    semantic_audit = (
        stage6_semantic_audit
        if isinstance(stage6_semantic_audit, dict)
        else {}
    )
    semantic_candidate = (
        stage6_semantic_candidate
        if isinstance(stage6_semantic_candidate, dict)
        else {}
    )

    semantic_validation = (
        semantic_summary.get("validation")
        or semantic_audit.get("validation")
        or {}
    )
    semantic_scientific_status = (
        semantic_summary.get("scientific_status")
        or semantic_audit.get("scientific_status")
        or {}
    )

    semantic_assist_view = {
        "enabled": bool(
            semantic_summary.get("enabled", False)
            or semantic_audit
            or semantic_candidate
        ),
        "executed": bool(
            semantic_summary.get("executed", False)
            or semantic_audit
            or semantic_candidate
        ),
        "status": (
            semantic_summary.get("status")
            or ("ok" if semantic_candidate else None)
        ),
        "return_code": semantic_summary.get("return_code"),
        "execution_profile": (
            stage6_semantic_state.get("execution_profile")
            if isinstance(stage6_semantic_state, dict)
            else None
        ),
        "validation": (
            semantic_validation
            if isinstance(semantic_validation, dict)
            else {}
        ),
        "scientific_status": (
            semantic_scientific_status
            if isinstance(semantic_scientific_status, dict)
            else {}
        ),
        "candidate": semantic_candidate,
        "audit_summary": {
            "artifact_version": semantic_audit.get("artifact_version"),
            "execution_strategy": semantic_audit.get("execution_strategy"),
            "runtime_seconds": semantic_audit.get("runtime_seconds"),
        },
    }

    run_gt_bundle = load_ground_truth_bundle(
        PROJECT_ROOT / "artifacts" / run_id
    )
    run_gt = run_gt_bundle.get("ground_truth")
    run_gt_provenance = run_gt_bundle.get("provenance")
    run_gt_benchmark = run_gt_bundle.get("stage5_benchmark")

    run_gt_available = bool(
        run_gt
        and run_gt_provenance
        and run_gt_benchmark
    )

    run_gt_summary = {
        "available": run_gt_available,
        "dataset_id": (
            run_gt.get("dataset_id")
            if isinstance(run_gt, dict)
            else None
        ),
        "source_text_status": (
            run_gt.get("source_text_status")
            if isinstance(run_gt, dict)
            else None
        ),
        "physical_line_alignment_status": (
            run_gt.get("physical_line_alignment_status")
            if isinstance(run_gt, dict)
            else None
        ),
        "translation_ground_truth_available": bool(
            run_gt.get(
                "translation_ground_truth_available",
                False,
            )
            if isinstance(run_gt, dict)
            else False
        ),
    }

    stage6_view = {
        "available": bool(
            stage6e
            or stage6f
            or semantic_candidate
            or semantic_audit
        ),
        "signals": {
            "H": readiness_page.get("htr_readiness_H_page"),
            "T": stage6f_metrics.get("page_T"),
        },
        "stage6": {
            "trust_status": stage6f_metrics.get("page_T_status"),
            "unresolved_lines": stage6f_metrics.get("unresolved_lines"),
            "adaptive_retry_required_lines": stage6f_metrics.get(
                "adaptive_retry_required_lines"
            ),
            "translation_eligible_lines": stage6f_metrics.get(
                "translation_eligible_lines"
            ),
            "reconstructed_lines": stage6e_metrics.get(
                "reconstructed_lines"
            ),
            "partially_supported_lines": stage6e_metrics.get(
                "partially_supported_lines"
            ),
            "abstained_lines": stage6e_metrics.get("abstained_lines"),
            "normalized_lines": stage6e_metrics.get("normalized_lines"),
            "reconstruction_evidence_index": stage6e_metrics.get(
                "page_reconstruction_evidence_index"
            ),
        },
        "adaptive_routing": (
            layer6_report if isinstance(layer6_report, dict) else {}
        ),
        "final_orchestration": (
            final_decision if isinstance(final_decision, dict) else {}
        ),
        "semantic_assist": semantic_assist_view,
    }

    return {
        "run_id": run_id,
        "presentation_contract": "generic-platform-v3-stage6-semantic-assist",
        "stage5": {
            "available": bool(readiness),
            "signals": {
                "H": readiness_page.get("htr_readiness_H_page")
            },
            "htr_readiness": {
                "H": readiness_page.get("htr_readiness_H_page"),
                "ground_truth_available": readiness_page.get(
                    "ground_truth_available",
                    False,
                ),
                "cer": readiness_page.get("cer"),
                "wer": readiness_page.get("wer"),
            },
            "provider_comparison": {
                "mean_content_char_similarity": comparison_aggregate.get(
                    "mean_content_char_similarity"
                ),
                "median_content_char_similarity": comparison_aggregate.get(
                    "median_content_char_similarity"
                ),
            },
            "transcription": transcription,
            "provider_c": {
                "available": bool(
                    manifest_c
                    or (
                        isinstance(provider_c_summary, dict)
                        and provider_c_summary.get("runtime_available", False)
                    )
                ),
                "executed": provider_c_executed,
                "provider_id": provider_meta.get("provider_id"),
                "model_id": provider_meta.get("model_id"),
                "page_display": provider_c_page_display or None,
                "lines": provider_c_lines,
            },
            "adaptive_evidence": adaptive_evidence,
            "ground_truth": run_gt_summary,
            "ground_truth_benchmark": (
                run_gt_benchmark
                if isinstance(run_gt_benchmark, dict)
                else {}
            ),
        },
        "stage6": stage6_view,
        "scientific_boundary": {
            "H_is_accuracy": False,
            "T_is_accuracy": False,
            "pairwise_H_is_accuracy": False,
            "provider_c_is_ground_truth": False,
            "translation_is_scholar_validated": False,
        },
    }

# GENERIC_RUN_GROUND_TRUTH_V1
@app.post("/pipeline/ground-truth/{run_id}")
def attach_generic_verified_ground_truth(run_id: str, payload: dict):
    """Attach externally verified transcription Ground Truth to an existing run."""
    run_dir = PROJECT_ROOT / "artifacts" / run_id

    if not run_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Unknown run_id", "run_id": run_id},
        )

    try:
        return attach_and_evaluate_ground_truth(
            run_id=run_id,
            run_dir=run_dir,
            payload=payload,
        )
    except GroundTruthValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Ground Truth validation failed",
                "detail": str(exc),
                "run_id": run_id,
            },
        )


@app.get("/pipeline/ground-truth/{run_id}")
def get_generic_verified_ground_truth(run_id: str):
    """Read-only status/benchmark view for run-level transcription Ground Truth."""
    run_dir = PROJECT_ROOT / "artifacts" / run_id

    if not run_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Unknown run_id", "run_id": run_id},
        )

    bundle = load_ground_truth_bundle(run_dir)
    benchmark = bundle.get("stage5_benchmark")

    return {
        "run_id": run_id,
        "ground_truth_available": bool(
            bundle.get("ground_truth")
            and bundle.get("provenance")
            and benchmark
        ),
        "ground_truth": bundle.get("ground_truth"),
        "provenance": bundle.get("provenance"),
        "stage5_benchmark": benchmark,
        "scientific_boundary": {
            "CER_WER_scope": "Stage-5 transcription only",
            "H_unchanged": True,
            "T_unchanged": True,
            "routing_thresholds_unchanged": True,
            "translation_validation_separate": True,
        },
    }
