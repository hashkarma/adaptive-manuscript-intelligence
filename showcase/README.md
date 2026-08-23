# Professor Showcase

This directory contains the research-showcase layer for the Adaptive
Manuscript Intelligence platform.

## Canonical structure

- `api/` — read-only FastAPI routes used by the professor dashboard.
- `cases/` — frozen, validated prior-execution evidence for the Golden and
  Difficult showcase cases.
- `config/` — execution profiles and showcase-specific configuration.
- reusable execution/reproduction scripts currently remain at the showcase
  root and will be moved into `tools/` only after path/import validation.
- local one-off diagnostics are stored under ignored `devtools/`.

## Runtime architecture

The showcase is not a separate application. It runs through the platform's
existing FastAPI entrypoint:

`backend.main:app`

The long-term repository remains organized around:

`backend -> frontend -> core -> layers -> orchestration -> stage runtimes`

with `models/`, `knowledge/`, `data/`, `artifacts/`, and `showcase/` acting as
supporting research/runtime assets.

## Scientific policy

- S, H and T are evidence/readiness/trust signals, not accuracy.
- CER/WER require verified ground truth.
- AI normalized text and translations remain candidates until scholar
  validation.
- Validated prior execution is visibly separated from live execution.

## Portability

Active showcase execution uses canonical inputs from:

`data/samples/golden_success/`

Generated runtime outputs belong under:

`artifacts/`

Frozen case JSON may contain repository-relative historical source-run paths.
Those fields are provenance metadata, not runtime dependencies.
