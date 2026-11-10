# PowerShell local startup script mimicking render_start.sh for Windows
$ErrorActionPreference = "Stop"

$env:PYTHONUNBUFFERED = "1"
$env:MLFLOW_TRACKING_URI = "file:./mlruns"
$env:API_BASE_URL = "http://127.0.0.1:8000"

Write-Host "Provisioning MLflow registry..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m src.models.provision_ci_registry

Write-Host "Starting FastAPI in background..." -ForegroundColor Cyan
$apiProcess = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -PassThru

try {
    Write-Host "Waiting for FastAPI readiness..." -ForegroundColor Cyan
    $ready = $false
    for ($i = 1; $i -le 30; $i++) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ready" -TimeoutSec 2 -ErrorAction Stop
            if ($resp.ready) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $ready) {
        Write-Error "FastAPI did not become ready in 30 seconds."
        exit 1
    }

    Write-Host "FastAPI is ready!" -ForegroundColor Green

    $port = if ($env:PORT) { $env:PORT } else { "8501" }
    Write-Host "Starting Streamlit on port $port..." -ForegroundColor Cyan
    & .\.venv\Scripts\streamlit.exe run src/app.py --server.port $port --server.address 0.0.0.0
} finally {
    Write-Host "Stopping background FastAPI process..." -ForegroundColor Yellow
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }
}

