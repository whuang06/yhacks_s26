#!/usr/bin/env bash
# Starts the FastAPI server and a static file server for the Tauri dev webview.
set -euo pipefail
DESKTOP="$(cd "$(dirname "$0")/.." && pwd)"
FRONT_END="$(cd "$DESKTOP/.." && pwd)"
REPO_ROOT="$(cd "$FRONT_END/.." && pwd)"

# Locate Python — prefer a venv at repo root or front_end level
PY=""
for candidate in "$REPO_ROOT/venv/bin/python" "$REPO_ROOT/.venv/bin/python" \
                  "$FRONT_END/venv/bin/python" "$FRONT_END/.venv/bin/python"; do
  if [[ -x "$candidate" ]]; then
    PY="$candidate"
    break
  fi
done
if [[ -z "$PY" ]]; then
  PY="$(command -v python3 2>/dev/null || true)"
fi
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "No Python found. Create a venv at $REPO_ROOT/venv and pip install -r $FRONT_END/requirements.txt" >&2
  exit 1
fi

# Load env vars (.env in desktop/, front_end/, or repo root)
for envfile in "$DESKTOP/.env" "$FRONT_END/.env" "$REPO_ROOT/.env"; do
  if [[ -f "$envfile" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$envfile"
    set +a
    echo "[dev.sh] Loaded $envfile"
  fi
done

# Start the FastAPI server from front_end/ so our server.py imports work.
# Restrict --reload to app + backend code only (avoid watching Tauri target/ and reload loops).
(cd "$FRONT_END" && "$PY" -m uvicorn server:app --host 127.0.0.1 --port 8765 --reload \
  --reload-dir "$FRONT_END" \
  --reload-dir "$REPO_ROOT/backend") &
UV_PID=$!
cleanup() { kill "$UV_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Serve the public/ directory for Tauri's devUrl
cd "$DESKTOP/public"
exec python3 -m http.server 1420 --bind 127.0.0.1
