#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# 1. Check if uv is installed and in PATH or ~/.local/bin
if command -v uv >/dev/null 2>&1; then
  echo "Starting backend server with uv..."
  exec uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
elif [ -f "$HOME/.local/bin/uv" ]; then
  echo "Starting backend server with ~/.local/bin/uv..."
  exec "$HOME/.local/bin/uv" run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
elif [ -d ".venv" ] && [ -f ".venv/bin/uvicorn" ]; then
  echo "Starting backend server with local .venv..."
  exec .venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
elif python3 -c "import uvicorn" >/dev/null 2>&1; then
  echo "Starting backend server with system python3..."
  exec python3 -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
else
  echo "Neither uv nor uvicorn could be found."
  echo "Please run:"
  echo "  uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload"
  exit 1
fi
