@echo off
REM ==============================================================================
REM  ENTERPRISE ML PIPELINE POC: 1-CLICK DEMO LAUNCHER
REM  Runs Airflow UI (8080) -> Executes ML Tasks -> Opens MLflow UI (5000)
REM  No Activate.ps1 or PowerShell execution policy required!
REM ==============================================================================

setlocal
cd /d "%~dp0"

echo.
echo ====================================================================
echo   NASH TECH AI POC: AUTOMATED DEMO WORKFLOW LAUNCHER
echo ====================================================================
echo.

if exist "ml-env\Scripts\python.exe" (
    set "PYTHON_EXE=ml-env\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" scripts\run_demo_flow.py

pause

