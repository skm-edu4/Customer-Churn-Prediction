#!/usr/bin/env bash
# Single-container deployment: FastAPI + Streamlit together.
#
# Used as the image's default CMD (Hugging Face Spaces runs ONE container on
# port 7860) and kept compatible with docker compose:
#
#   * API      → 127.0.0.1:8000  (loopback only; the dashboard calls it over
#                  HTTP, the public surface never exposes it directly)
#   * Dashboard → 0.0.0.0:${PORT:-7860}  (the one public entry point)
#
# Streamlit runs in the foreground as PID 1 so it receives SIGTERM from the
# platform; uvicorn is a child that dies with the container.
set -euo pipefail

API_PORT="${CHURN_INTERNAL_API_PORT:-8000}"
DASH_PORT="${PORT:-7860}"

export CHURN_API_URL="http://127.0.0.1:${API_PORT}"

/app/.venv/bin/uvicorn churn.serving.api:app \
    --host 127.0.0.1 \
    --port "${API_PORT}" \
    --log-level warning &
API_PID=$!

term() {
    kill "${API_PID}" 2>/dev/null || true
}
trap term TERM INT

exec /app/.venv/bin/streamlit run /app/dashboard/app.py \
    --server.address 0.0.0.0 \
    --server.port "${DASH_PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false
