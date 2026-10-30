"""Master Demo Flow: Orchestrates the entire POC presentation sequence.

Execution Sequence:
  Phase 1: Launches Apache Airflow UI at http://localhost:8080 and opens it in browser.
  Phase 2: Executes the Machine Learning Tasks (Spark -> Sklearn -> PyTorch -> Evaluation).
  Phase 3: Launches MLflow Tracking Server at http://localhost:5000 and opens it in browser.

Usage (No Activate.ps1 needed!):
  .\\ml-env\\Scripts\\python.exe scripts/run_demo_flow.py
"""

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from src.ingestion.spark_processor import configure_java_home
from src.utils import get_logger, load_config


def wait_for_port(url: str, timeout: int = 15) -> bool:
    """Waits until a local HTTP service responds with status 200."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status in (200, 302, 404):
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    logger = get_logger("MasterDemoFlow")
    python_exe = sys.executable

    print("\n" + "=" * 70)
    print("   ENTERPRISE ML PIPELINE POC: FULL DEMO PRESENTATION SEQUENCE")
    print("=" * 70)
    print("  Phase 1: Launch Apache Airflow DAG Dashboard (Port 8080)")
    print("  Phase 2: Execute Machine Learning Tasks (Spark -> Sklearn -> PyTorch)")
    print("  Phase 3: Launch MLflow Tracking Dashboard (Port 5000)")
    print("=" * 70 + "\n")

    # Ensure Java 17 is targeted
    configure_java_home(logger)

    # -------------------------------------------------------------
    # PHASE 1: Start Airflow UI First
    # -------------------------------------------------------------
    logger.info(">>> [PHASE 1/3] Starting Apache Airflow UI Server on port 8080...")
    airflow_script = os.path.join("scripts", "start_airflow_ui.py")
    airflow_proc = subprocess.Popen([python_exe, airflow_script])

    # Wait for server to respond and open in browser
    if wait_for_port("http://localhost:8080"):
        logger.info("Airflow UI is LIVE! Opening browser: http://localhost:8080")
        webbrowser.open("http://localhost:8080")
    else:
        logger.warning("Airflow UI starting slowly, opening browser anyway...")
        webbrowser.open("http://localhost:8080")

    time.sleep(2)

    # -------------------------------------------------------------
    # PHASE 2: Start Machine Learning Tasks
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    logger.info(">>> [PHASE 2/3] Starting Machine Learning Pipeline Tasks locally...")
    print("-" * 70)

    pipeline_script = os.path.join("scripts", "run_pipeline_local.py")
    cmd = [python_exe, pipeline_script, "--config", "config/development.yaml"]
    
    pipeline_exit_code = subprocess.call(cmd)
    if pipeline_exit_code != 0:
        logger.error(f"Pipeline tasks failed with exit code {pipeline_exit_code}!")
        sys.exit(pipeline_exit_code)

    logger.info("Machine Learning Pipeline completed successfully!")

    # -------------------------------------------------------------
    # PHASE 3: Launch MLflow Tracking Server
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    logger.info(">>> [PHASE 3/3] Starting MLflow Tracking Server on port 5000...")
    print("-" * 70)

    # Path to mlflow executable inside the virtualenv
    venv_bin_dir = os.path.dirname(python_exe)
    mlflow_exe = os.path.join(venv_bin_dir, "mlflow.exe" if os.name == "nt" else "mlflow")
    if not os.path.exists(mlflow_exe):
        mlflow_exe = "mlflow"

    mlflow_cmd = [
        mlflow_exe,
        "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    mlflow_proc = subprocess.Popen(mlflow_cmd)

    if wait_for_port("http://localhost:5000"):
        logger.info("MLflow Server is LIVE! Opening browser: http://localhost:5000")
        webbrowser.open("http://localhost:5000")
    else:
        logger.warning("MLflow Server starting, opening browser...")
        webbrowser.open("http://localhost:5000")

    # -------------------------------------------------------------
    # ALL SERVICES READY
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("   ALL DEMO SERVICES & ML RESULTS ARE LIVE!")
    print("=" * 70)
    print("  1. Airflow Orchestration UI : http://localhost:8080")
    print("  2. MLflow Experiment UI     : http://localhost:5000")
    print("  3. PyTorch Model Checkpoint : models/fare_model.pt")
    print("  4. Sklearn Preprocessor     : models/preprocessor.joblib")
    print("  5. Diagnostic Plots         : models/actual_vs_predicted.png")
    print("=" * 70)
    print("Press Ctrl+C in this terminal when you want to stop the demo servers.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down demo servers...")
        airflow_proc.terminate()
        mlflow_proc.terminate()
        print("Demo servers stopped cleanly.")


if __name__ == "__main__":
    main()

