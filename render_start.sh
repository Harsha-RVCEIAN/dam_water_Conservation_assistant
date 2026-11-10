#!/usr/bin/env bash
set -e

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="file:./mlruns"
export API_BASE_URL="http://127.0.0.1:8000"

API_PID=""
STREAMLIT_PID=""

cleanup() {
  echo "Stopping application processes..."
  if [ -n "$STREAMLIT_PID" ]; then
    kill "$STREAMLIT_PID" 2>/dev/null || true
  fi
  if [ -n "$API_PID" ]; then
    kill "$API_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Provisioning MLflow registry..."
python -m src.models.provision_ci_registry

echo "Starting FastAPI..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

echo "Waiting for FastAPI readiness..."
READY=0
for i in $(seq 1 30); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: FastAPI did not become ready."
  exit 1
fi

echo "FastAPI is ready."

PORT="${PORT:-10000}"
echo "Starting Streamlit on port $PORT..."

streamlit run src/app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true &
STREAMLIT_PID=$!

wait "$STREAMLIT_PID"
