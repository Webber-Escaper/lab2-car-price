#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
exec "$SCRIPT_DIR/../.venv_lab1/bin/mlflow" server \
  --backend-store-uri "sqlite:///$SCRIPT_DIR/mlruns.db" \
  --serve-artifacts \
  --artifacts-destination "$SCRIPT_DIR/mlartifacts" \
  --host 127.0.0.1 \
  --port 5000
