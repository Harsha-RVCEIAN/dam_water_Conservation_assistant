# Multi-stage production container for KRS Dam Water Conservation Assistant
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    LOKY_MAX_CPU_COUNT=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and assets
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/interim/selected_rainfall_stations.csv /app/data/interim/selected_rainfall_stations.csv
COPY data/processed/krs_dam_processed.csv /app/data/processed/krs_dam_processed.csv
COPY reports/ /app/reports/

# Create cache directories with appropriate permissions
RUN mkdir -p /app/.cache/matplotlib /app/data/processed /app/data/interim

# Default service: FastAPI Inflow Prediction API (Port 8000)
EXPOSE 8000 8501 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
