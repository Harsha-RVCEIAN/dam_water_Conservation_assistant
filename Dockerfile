# Multi-stage production Dockerfile for K.R.Sagara Dam Inflow Forecasting API
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install security updates and curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies including production serving stack
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi \
    "uvicorn[standard]" \
    httpx

# Copy application source code, models, and necessary processed data
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/
COPY reports/ /app/reports/

# Create non-root user for principle of least privilege
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose FastAPI default port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server entrypoint
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
