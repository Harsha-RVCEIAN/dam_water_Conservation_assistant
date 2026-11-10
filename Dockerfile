# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables for clean container execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MLFLOW_TRACKING_URI="file:/app/mlruns"

# Set application working directory
WORKDIR /app

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and runtime metadata
COPY src/ ./src/
COPY models/model_metadata.json ./models/model_metadata.json
COPY config/ ./config/

# Expose HTTP inference port
EXPOSE 8000

# Container liveness health check using Python standard library (no curl/wget required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Run FastAPI serving application via Uvicorn bound to 0.0.0.0
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
