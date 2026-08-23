# AWS Full-Pipeline Deployment

Validated deployment target: **Ubuntu 24.04 LTS**, Python 3.12.

This deployment preserves the existing FastAPI entrypoint:

```bash
PYTHONPATH=. python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The EC2 host recreates the local environment boundaries:

- `.venv` — FastAPI, Stage 0-4, Bedrock/Qwen Provider C
- `venv-stage5` — Provider A
- `venv-provider-b` — Provider B
- `venv-stage6` — Vidyut deterministic Stage 6
- `venv-stage6-rag` — RapidFuzz Stage 6D.2

Mac virtual environments are never copied.

## Bootstrap order

```bash
chmod +x deploy/aws/*.sh

deploy/aws/bootstrap_base.sh
deploy/aws/bootstrap_stage5.sh
deploy/aws/bootstrap_stage6.sh
```

Provision Stage-6 assets:

```bash
export STAGE6_ASSET_S3_URI=s3://YOUR-BUCKET/manuscript-intelligence/stage6-assets.tar.gz
deploy/aws/bootstrap_stage6_assets.sh
```

Verify:

```bash
deploy/aws/verify_full_pipeline.sh
```

Verify EC2-role → Bedrock token flow:

```bash
VERIFY_BEDROCK_TOKEN=1 deploy/aws/verify_full_pipeline.sh
```

Optional live Qwen inference smoke test:

```bash
VERIFY_BEDROCK_TOKEN=1 \
VERIFY_BEDROCK_INFERENCE=1 \
deploy/aws/verify_full_pipeline.sh
```

Start platform:

```bash
deploy/aws/run_platform.sh
```

The existing application continues to expose generic upload and Stage 0→6
routes; the Golden and Difficult showcase cases remain separate validated
research demonstrations.

## Stage-6 asset packaging

Run this from the research workstation after AWS CLI access is configured:

```bash
export STAGE6_ASSET_S3_URI=s3://YOUR-BUCKET/manuscript-intelligence/stage6-assets.tar.gz
deploy/aws/package_stage6_assets.sh
```

The script creates a tarball, computes SHA-256, and uploads both the archive and
checksum. EC2 verifies the checksum before extraction.
