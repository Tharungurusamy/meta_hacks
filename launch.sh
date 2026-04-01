#!/usr/bin/env bash
# One-command local run: dependencies + FastAPI (CyberSec-OpenEnv).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

if ! python3 -c "import fastapi, uvicorn, openenv, pydantic" 2>/dev/null; then
  echo "[launch] Installing Python dependencies from requirements.txt ..."
  if ! python3 -m pip install -r requirements.txt -q 2>/dev/null; then
    python3 -m pip install -r requirements.txt --break-system-packages -q
  fi
fi

echo "[launch] Starting http://${HOST}:${PORT}/  (UI: /  and /web , API: /docs)"
exec python3 -m uvicorn server.app:app --host "$HOST" --port "$PORT"
