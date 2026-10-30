# PowerShell script to launch both Airflow UI and MLflow Server for Windows Live Demos
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " LAUNCHING POC LIVE DEMO SERVICES (AIRFLOW + MLFLOW)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# 1. Start Airflow UI on Port 8080 (Direct python.exe, no Activate.ps1 needed)
Write-Host "[1/3] Starting Airflow DAG Dashboard on http://localhost:8080..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; .\ml-env\Scripts\python.exe scripts/start_airflow_ui.py"

# 2. Start MLflow Tracking Server on Port 5000 (Direct mlflow.exe)
Write-Host "[2/3] Starting MLflow Tracking Server on http://localhost:5000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; .\ml-env\Scripts\mlflow.exe server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000"

Start-Sleep -Seconds 2

# 3. Open both dashboards in default browser
Write-Host "[3/3] Opening browser dashboards..." -ForegroundColor Green
Start-Process "http://localhost:8080"
Start-Process "http://localhost:5000"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " All Demo Services are LIVE!" -ForegroundColor Green
Write-Host "   - Airflow UI : http://localhost:8080" -ForegroundColor White
Write-Host "   - MLflow UI  : http://localhost:5000" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "To execute the pipeline locally, run:" -ForegroundColor Yellow
Write-Host "  python scripts/run_pipeline_local.py --config config/development.yaml" -ForegroundColor White
Write-Host "(Or simply click 'Trigger DAG Run' directly on http://localhost:8080!)" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
