#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVICE_NAME="manuscript-intelligence"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
RUN_USER="${MANUSCRIPT_SERVICE_USER:-$(id -un)}"
RUN_GROUP="${MANUSCRIPT_SERVICE_GROUP:-$(id -gn)}"

[[ -x "$REPO_ROOT/deploy/aws/run_platform.sh" ]] || {
  echo "ERROR: run_platform.sh is not executable."
  exit 1
}

TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

cat > "$TMP_FILE" <<EOF
[Unit]
Description=Adaptive Manuscript Intelligence Platform
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_GROUP
WorkingDirectory=$REPO_ROOT
Environment=AWS_REGION=us-east-1
Environment=AWS_DEFAULT_REGION=us-east-1
Environment=PYTHONUNBUFFERED=1
ExecStart=$REPO_ROOT/deploy/aws/run_platform.sh
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo install -m 0644 "$TMP_FILE" "$SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo
sudo systemctl --no-pager --full status "$SERVICE_NAME" || true
echo
echo "Installed: $SERVICE_FILE"
