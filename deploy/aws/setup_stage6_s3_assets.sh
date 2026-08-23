#!/usr/bin/env bash
set -euo pipefail

PROFILE="${AWS_PROFILE:-aws-showcase-demo}"
REGION="${AWS_REGION:-us-east-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ARCHIVE="artifacts/deployment/stage6-assets.tar.gz"
CHECKSUM="${ARCHIVE}.sha256"

[[ -f "$ARCHIVE" ]] || {
  echo "ERROR: Missing $ARCHIVE"
  exit 1
}
[[ -f "$CHECKSUM" ]] || {
  echo "ERROR: Missing $CHECKSUM"
  exit 1
}

ACCOUNT_ID="$(
  aws sts get-caller-identity \
    --profile "$PROFILE" \
    --query Account \
    --output text
)"

BUCKET="${STAGE6_ASSET_BUCKET:-adaptive-manuscript-assets-${ACCOUNT_ID}}"
KEY_PREFIX="${STAGE6_ASSET_PREFIX:-manuscript-intelligence/runtime/stage6}"
OBJECT_KEY="${KEY_PREFIX}/stage6-assets.tar.gz"
S3_URI="s3://${BUCKET}/${OBJECT_KEY}"

echo "======================================================================"
echo "Adaptive Manuscript Intelligence — Stage-6 S3 Asset Setup"
echo "======================================================================"
echo "AWS profile : $PROFILE"
echo "AWS region  : $REGION"
echo "Bucket      : $BUCKET"
echo "Object      : $OBJECT_KEY"
echo

if aws s3api head-bucket \
  --bucket "$BUCKET" \
  --profile "$PROFILE" \
  >/dev/null 2>&1; then
  echo "[1/5] Bucket already exists and is accessible."
else
  echo "[1/5] Creating private S3 bucket..."
  if [[ "$REGION" == "us-east-1" ]]; then
    aws s3api create-bucket \
      --bucket "$BUCKET" \
      --region "$REGION" \
      --profile "$PROFILE" \
      >/dev/null
  else
    aws s3api create-bucket \
      --bucket "$BUCKET" \
      --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=${REGION}" \
      --profile "$PROFILE" \
      >/dev/null
  fi
fi

echo "[2/5] Blocking all public access..."
aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --profile "$PROFILE" \
  --public-access-block-configuration \
'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'

echo "[3/5] Enabling bucket versioning..."
aws s3api put-bucket-versioning \
  --bucket "$BUCKET" \
  --profile "$PROFILE" \
  --versioning-configuration Status=Enabled

echo "[4/5] Enabling default server-side encryption..."
aws s3api put-bucket-encryption \
  --bucket "$BUCKET" \
  --profile "$PROFILE" \
  --server-side-encryption-configuration \
'{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'

echo "[5/5] Uploading Stage-6 archive and checksum..."
aws s3 cp "$ARCHIVE" "$S3_URI" \
  --profile "$PROFILE" \
  --region "$REGION" \
  --sse AES256

aws s3 cp "$CHECKSUM" "${S3_URI}.sha256" \
  --profile "$PROFILE" \
  --region "$REGION" \
  --sse AES256

echo
echo "Verifying uploaded objects..."
aws s3api head-object \
  --bucket "$BUCKET" \
  --key "$OBJECT_KEY" \
  --profile "$PROFILE" \
  --query '{ContentLength:ContentLength,ETag:ETag,VersionId:VersionId,ServerSideEncryption:ServerSideEncryption}'

aws s3api head-object \
  --bucket "$BUCKET" \
  --key "${OBJECT_KEY}.sha256" \
  --profile "$PROFILE" \
  --query '{ContentLength:ContentLength,ETag:ETag,VersionId:VersionId,ServerSideEncryption:ServerSideEncryption}'

mkdir -p showcase/devtools/reports

cat > showcase/devtools/reports/stage6_s3_asset_location.txt <<EOF
STAGE6_ASSET_S3_URI=${S3_URI}
STAGE6_ASSET_BUCKET=${BUCKET}
STAGE6_ASSET_REGION=${REGION}
LOCAL_SHA256=$(cat "$CHECKSUM")
EOF

echo
echo "======================================================================"
echo "STAGE-6 S3 ASSET SETUP: PASS"
echo "======================================================================"
echo "STAGE6_ASSET_S3_URI=$S3_URI"
echo
echo "Saved:"
echo "showcase/devtools/reports/stage6_s3_asset_location.txt"
echo "======================================================================"
