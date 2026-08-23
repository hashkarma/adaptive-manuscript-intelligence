# Runtime Asset Strategy

The Git repository is the source of truth for application code, configuration,
tests, frozen showcase evidence, and deployment automation.

Large runtime assets are intentionally not treated as normal Git source files.

## Stage-6 assets used by the local research pipeline

- `models/vidyut-0.4.0/` — Sanskrit linguistic/model resources.
- `knowledge/stage6d_dcs/passages.jsonl` — deterministic contextual retrieval
  corpus.

These assets are required for a full deterministic Stage-6 rerun, but they are
not required for serving the frozen professor-showcase dashboard.

For AWS, provision large runtime assets separately (for example from a
versioned S3 research-assets bucket) and verify them before starting full
Stage-6 execution. EC2 must not receive Mac virtual environments.

## Showcase execution policy

- The Golden and Difficult frozen cases under `showcase/cases/` remain
  reviewable without these large runtime assets.
- Selected live Stage 0-4 and Bedrock/Qwen checks can run independently.
- Full Stage-6 deterministic execution requires the provisioned assets above.
