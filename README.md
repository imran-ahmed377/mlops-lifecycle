# Simple Iris MLOps Project

This project is a very small MLOps example built around the Iris dataset and a Logistic Regression model.

The goal is to keep the project easy to understand while introducing these tools step by step:
- DVC for data and model versioning
- MLflow for experiment and model tracking
- Prometheus for monitoring and drift metrics
- Streamlit for a simple real-time prediction UI

## Current Status

Implemented so far:
- reproducible data preparation
- model training with MLflow logging
- standalone evaluation step
- local Git + DVC repository initialized
- DVC pipeline stages for `prepare`, `train`, `evaluate`, and `retrain`
- FastAPI prediction service with `/health`, `/predict`, `/metrics`, `/drift/status`, and `/drift/check`
- Prometheus scrape configuration for API monitoring
- Prometheus alert rules for drift and model availability
- Streamlit frontend for live prediction and drift visibility
- KS-based drift monitoring using baseline vs live request data
- manual retraining with a promotion gate that only keeps the candidate model if it is not worse than the current model

## Project Structure

```text
app/
data/
drift/
metrics/
models/
prometheus/
src/
tests/
requirements.txt
params.yaml
dvc.yaml
```

## Quick Start

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the pipeline manually

Prepare data:

```powershell
python -m src.data_prep
```

Train model:

```powershell
python -m src.train
```

Evaluate model:

```powershell
python -m src.evaluate
```

### 4. Run the reproducible DVC pipeline

The DVC pipeline is already initialized. Run the main stages with:

```powershell
dvc repro
```

To run only retraining:

```powershell
dvc repro retrain
```

### 5. Run Prometheus for monitoring

Keep the API running first so Prometheus can scrape `http://127.0.0.1:8000/metrics`.

The Prometheus configuration is stored in [prometheus/prometheus.yml](prometheus/prometheus.yml) and it loads alert rules from [prometheus/alerts.yml](prometheus/alerts.yml).

Ready-to-use PromQL examples for a first dashboard are listed in [prometheus/dashboard_queries.md](prometheus/dashboard_queries.md).

If `prometheus.exe` is available on your machine, run:

```powershell
prometheus.exe --config.file=prometheus/prometheus.yml
```

If Prometheus is added to your PATH, this also works:

```powershell
prometheus --config.file=prometheus/prometheus.yml
```

Then open:

```text
http://127.0.0.1:9090
```

Current alert rules:
- `IrisDataDriftAlert`: drift status reached alert level and stayed there for 1 minute.
- `IrisDataDriftWarning`: drift status reached warning level and stayed there for 2 minutes.
- `IrisDriftSignalMissing`: drift signal is still unavailable or there is not enough live data for 5 minutes.
- `IrisModelUnavailable`: the API does not have a trained model loaded for 1 minute.

### 6. Run the prediction API

```powershell
python -m src.serve_api
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 7. Run the Streamlit UI

```powershell
python -m streamlit run app/streamlit_app.py --server.headless true --server.port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

### 8. Run drift check manually

```powershell
python -m src.drift_monitor
```

### 9. Run retraining manually

```powershell
python -m src.retrain
```

## Outputs Created So Far

- raw dataset: `data/raw/iris.csv`
- train split: `data/processed/train.csv`
- validation split: `data/processed/valid.csv`
- trained model: `models/model.joblib`
- evaluation metrics: `metrics/eval_metrics.json`
- confusion matrix: `metrics/confusion_matrix.csv`
- drift baseline reference: `drift/baseline_reference.csv`
- drift baseline stats: `drift/baseline_stats.json`
- live prediction log: `drift/live_requests.csv`
- latest drift status: `drift/drift_status.json`
- retraining report: `metrics/retrain_report.json`
- local dependency compatibility report: `python_package_compatibility.txt`

## Next Steps

The next implementation steps are:
- add a simple Prometheus or Grafana dashboard for live metrics
- replace pseudo-label retraining with real labeled feedback data
- add automated tests for API, drift logic, and retraining gate
- add API smoke tests for `/predict`, `/drift/status`, and `/metrics`
- optionally connect the local Git repo to GitHub for backup and collaboration