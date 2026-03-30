#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
DEPS_STAMP="$VENV_DIR/.deps_installed"
HOST="127.0.0.1"
PORT="8000"
APP_URL="http://$HOST:$PORT"

cd "$PROJECT_ROOT"

echo "[1/5] Preparing virtual environment..."
if [[ ! -d "$VENV_DIR" ]]; then
	python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] Checking dependencies..."
if [[ ! -f "$DEPS_STAMP" || requirements.txt -nt "$DEPS_STAMP" || frontend/requirements_frontend.txt -nt "$DEPS_STAMP" ]]; then
	echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -r frontend/requirements_frontend.txt
	touch "$DEPS_STAMP"
else
	echo "Dependencies already up to date."
fi

echo "[3/5] Starting FastAPI server..."
python -m uvicorn frontend.main:app --host "$HOST" --port "$PORT" --app-dir "$PROJECT_ROOT" &
SERVER_PID=$!

cleanup() {
	if kill -0 "$SERVER_PID" 2>/dev/null; then
		kill "$SERVER_PID" 2>/dev/null || true
	fi
}
trap cleanup EXIT INT TERM

echo "[4/5] Waiting for server readiness..."
for _ in {1..100}; do
	if curl -fsS "$APP_URL" >/dev/null 2>&1; then
		break
	fi
	sleep 0.2
done

echo "[5/5] Opening web app: $APP_URL"
if command -v open >/dev/null 2>&1; then
	open "$APP_URL" >/dev/null 2>&1 || true
fi

echo "Server running. Press Ctrl+C to stop."
wait "$SERVER_PID"
