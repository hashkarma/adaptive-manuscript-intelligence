from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/showcase", tags=["showcase"])

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPO_ROOT / "showcase" / "cases"
CATALOG_PATH = CASES_ROOT / "showcase_catalog.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Showcase artifact not found: {path.name}",
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid showcase JSON: {path.name}: {exc}",
        ) from exc

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Expected JSON object in showcase artifact: {path.name}",
        )

    return data


def _catalog() -> Dict[str, Any]:
    return _load_json(CATALOG_PATH)


def _case_entry(case_id: str) -> Dict[str, Any]:
    catalog = _catalog()
    cases = catalog.get("cases", [])

    if not isinstance(cases, list):
        raise HTTPException(
            status_code=500,
            detail="Showcase catalog 'cases' field is invalid.",
        )

    for case in cases:
        if isinstance(case, dict) and case.get("id") == case_id:
            return case

    raise HTTPException(
        status_code=404,
        detail=f"Unknown showcase case: {case_id}",
    )


@router.get("/health")
def showcase_health() -> Dict[str, Any]:
    catalog = _catalog()

    cases = catalog.get("cases", [])
    stage_model = catalog.get("stage_model", [])
    modes = catalog.get("available_execution_modes", [])

    return {
        "status": "ok",
        "service": "adaptive-manuscript-intelligence-showcase",
        "catalog_version": catalog.get("catalog_version"),
        "case_count": len(cases) if isinstance(cases, list) else 0,
        "stage_count": len(stage_model) if isinstance(stage_model, list) else 0,
        "execution_mode_count": len(modes) if isinstance(modes, list) else 0,
        "catalog_path": str(CATALOG_PATH),
    }


@router.get("/catalog")
def get_showcase_catalog() -> Dict[str, Any]:
    return _catalog()


@router.get("/cases")
def list_showcase_cases() -> Dict[str, Any]:
    catalog = _catalog()
    cases = catalog.get("cases", [])

    if not isinstance(cases, list):
        raise HTTPException(
            status_code=500,
            detail="Showcase catalog 'cases' field is invalid.",
        )

    return {
        "count": len(cases),
        "cases": cases,
    }


@router.get("/cases/{case_id}")
def get_showcase_case(case_id: str) -> Dict[str, Any]:
    return _case_entry(case_id)


@router.get("/cases/{case_id}/ui-state")
def get_showcase_case_ui_state(case_id: str) -> Dict[str, Any]:
    case = _case_entry(case_id)
    rel = case.get("ui_state_file")

    if not isinstance(rel, str) or not rel.strip():
        raise HTTPException(
            status_code=500,
            detail=f"UI state file is not declared for case: {case_id}",
        )

    return _load_json(CASES_ROOT / rel)


@router.get("/cases/{case_id}/summary")
def get_showcase_case_summary(case_id: str) -> Dict[str, Any]:
    case = _case_entry(case_id)
    rel = case.get("summary_file")

    if not isinstance(rel, str) or not rel.strip():
        raise HTTPException(
            status_code=500,
            detail=f"Summary file is not declared for case: {case_id}",
        )

    return _load_json(CASES_ROOT / rel)
