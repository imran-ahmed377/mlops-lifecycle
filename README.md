# MLOps Lifecycle: Iris Classification

A compact, end-to-end MLOps reference project built on the Iris dataset.
It demonstrates how to move from reproducible data preparation to model serving,
drift monitoring, and promotion-gated retraining with a practical local stack.

## What This Project Covers

- Reproducible data preparation and train/validation split
- Logistic Regression training with MLflow experiment logging
- Offline evaluation with metrics and confusion matrix artifacts
- FastAPI inference service with Prometheus metrics exposure
- Drift detection with Kolmogorov-Smirnov (KS) statistics
- Retraining workflow with a promotion gate (candidate must not regress)
- Streamlit dashboard for live predictions and drift visibility

## Architecture Overview

1. `src.data_prep` loads Iris data, creates train/validation splits, and builds drift baseline artifacts.
2. `src.train` trains a scikit-learn pipeline and logs run metadata to MLflow.
3. `src.evaluate` writes validation metrics and confusion matrix outputs.
4. `src.serve_api` serves `/predict`, logs live request features, and exposes `/metrics`.
5. `src.drift_monitor` compares baseline vs live feature distributions using KS tests.
6. `src.retrain` trains a candidate model and promotes it only when gate criteria are met.

## Technology Stack

- Python, pandas, scikit-learn, SciPy
- DVC for pipeline reproducibility
- MLflow for experiment tracking and model artifacts
- FastAPI + Uvicorn for inference serving
- Prometheus for metrics scraping and alerting
- Streamlit for interactive UI

## Repository Layout

```text
app/                    # Streamlit app
data/                   # Raw and processed datasets
drift/                  # Drift baseline, live requests, drift status
metrics/                # Evaluation and retraining reports
models/                 # Active model artifact
prometheus/             # Prometheus config, alerts, query examples
src/                    # Pipeline and service source code
tests/                  # Unit tests
dvc.yaml                # Reproducible pipeline stages
params.yaml             # Centralized configuration
requirements.txt        # Python dependencies
```

## Quick Start (Windows / PowerShell)

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If execution policy blocks activation:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the full reproducible pipeline

```powershell
dvc repro
```

### 4. Start the inference API

```powershell
python -m src.serve_api
```

Default API endpoint: http://127.0.0.1:8000

### 5. Start the Streamlit app

```powershell
python -m streamlit run app/streamlit_app.py --server.headless true --server.port 8501
```

Streamlit URL: http://127.0.0.1:8501

### 6. Start Prometheus (optional but recommended)

```powershell
prometheus --config.file=prometheus/prometheus.yml
```

Prometheus URL: http://127.0.0.1:9090

## Useful Commands

Run stages manually:

```powershell
python -m src.data_prep
python -m src.train
python -m src.evaluate
python -m src.drift_monitor
python -m src.retrain
```

Run only retraining stage with DVC:

```powershell
dvc repro retrain
```

Run unit tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Start MLflow UI against the same local tracking backend used by training:

```powershell
$trackingUri = python -c "import mlflow; print(mlflow.get_tracking_uri())"
python -m mlflow ui --backend-store-uri $trackingUri --port 5000
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service readiness and model-loaded flag |
| POST | `/predict` | Predict iris class for a single feature payload |
| GET | `/metrics` | Prometheus-formatted metrics |
| GET | `/drift/status` | Return most recent drift assessment |
| POST | `/drift/check` | Trigger a fresh drift check and return result |

## Key Artifacts

| Path | Description |
|---|---|
| `data/raw/iris.csv` | Raw Iris dataset export |
| `data/processed/train.csv` | Training split |
| `data/processed/valid.csv` | Validation split |
| `models/model.joblib` | Currently active model |
| `metrics/eval_metrics.json` | Validation metrics report |
| `metrics/confusion_matrix.csv` | Confusion matrix output |
| `drift/baseline_reference.csv` | Baseline feature reference for drift checks |
| `drift/baseline_stats.json` | Baseline feature summary statistics |
| `drift/live_requests.csv` | Logged live prediction feature rows |
| `drift/drift_status.json` | Latest drift result |
| `metrics/retrain_report.json` | Retraining and promotion decision report |

## Monitoring and Alerts

Prometheus scraping and alert rules are defined in:

- `prometheus/prometheus.yml`
- `prometheus/alerts.yml`
- `prometheus/dashboard_queries.md`

Implemented alert rules:

- `IrisDataDriftAlert` (critical)
- `IrisDataDriftWarning` (warning)
- `IrisDriftSignalMissing` (info)
- `IrisModelUnavailable` (critical)

## Experiment Tracking

- Training and retraining runs are logged with MLflow.
- Local MLflow artifacts are stored under `mlruns/`.
- Promoted model artifacts are logged during successful retraining runs.

## Notes on Retraining Gate

The candidate model is promoted only if both validation metrics are at least as good as the currently deployed model:

- `valid_accuracy`
- `valid_f1_macro`

If no current model exists, the candidate is promoted by default.

## Roadmap

- Add automated CI for tests and pipeline validation
- Introduce real labeled feedback instead of pseudo-label retraining
- Add model registry and release tagging workflow
- Add dashboard snapshots and operational runbooks