from __future__ import annotations  # Allow postponed evaluation of type annotations.

import csv  # Append live prediction rows to a CSV log.
import json  # Read baseline and drift-status JSON files.
from pathlib import Path  # Handle filesystem paths safely.
from time import perf_counter  # Measure request latency.
from typing import Dict, List  # Type hints for mappings and feature lists.

import joblib  # Load the trained model artifact from disk.
import pandas as pd  # Build DataFrames for model inference.
import uvicorn  # Run the FastAPI application locally.
from fastapi import FastAPI, HTTPException  # Define API routes and HTTP errors.
from pydantic import BaseModel  # Validate request payloads.
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest  # Expose Prometheus metrics.
from starlette.responses import Response  # Return raw metrics payloads.

from src.common import ensure_parent_dir, load_params  # Shared helpers for config and directory creation.
from src.drift_monitor import run_drift_check  # Reuse the batch drift checker from the monitoring module.

FEATURE_MAPPING: Dict[str, str] = {
    "sepal_length_cm": "sepal length (cm)",  # API field name -> model feature name.
    "sepal_width_cm": "sepal width (cm)",  # API field name -> model feature name.
    "petal_length_cm": "petal length (cm)",  # API field name -> model feature name.
    "petal_width_cm": "petal width (cm)",  # API field name -> model feature name.
}

CLASS_MAPPING: Dict[int, str] = {
    0: "setosa",  # Class ID 0 maps to setosa.
    1: "versicolor",  # Class ID 1 maps to versicolor.
    2: "virginica",  # Class ID 2 maps to virginica.
}

# Count prediction requests and label them by endpoint and outcome.
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests.",
    ["endpoint", "status"],
)

# Track end-to-end latency for prediction requests.
PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds.",
)

# Count how often each class is predicted.
PREDICTION_CLASS_TOTAL = Counter(
    "prediction_class_total",
    "Count of predictions by class.",
    ["class_id"],
)

# Report whether a model artifact is currently loaded into memory.
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether a model artifact is loaded (1) or not (0).",
)

# Store the latest overall drift status as a numeric value for alerting.
DRIFT_STATUS = Gauge(
    "data_drift_status",
    "Current drift state as numeric value (ok=0, warning=1, alert=2, insufficient_data=-1).",
)

# Track the largest KS score from the latest drift run.
DRIFT_MAX_KS = Gauge(
    "data_drift_max_ks_stat",
    "Maximum KS drift statistic across monitored features from latest check.",
)

# Track per-feature KS statistics from the latest drift run.
DRIFT_FEATURE_KS = Gauge(
    "data_drift_feature_ks_stat",
    "KS drift statistic for each monitored feature from latest check.",
    ["feature"],
)


class PredictRequest(BaseModel):
    """Validated request body for a single iris prediction."""

    sepal_length_cm: float  # Sepal length supplied by the client.
    sepal_width_cm: float  # Sepal width supplied by the client.
    petal_length_cm: float  # Petal length supplied by the client.
    petal_width_cm: float  # Petal width supplied by the client.


class ModelService:
    """Manage model loading, feature ordering, and live-request logging."""

    def __init__(self) -> None:
        self.params = load_params()  # Load configuration once when the service starts.
        self.paths = self.params["paths"]  # Cache configured artifact paths.
        self.model = None  # Model instance is loaded separately at startup.
        self.feature_columns = self._load_feature_columns()  # Preserve model feature order.

    def _load_feature_columns(self) -> List[str]:
        """Load feature order from baseline stats or fall back to defaults."""

        baseline_stats_path = Path(self.paths["baseline_stats"])  # Location of baseline metadata created during data prep.
        if baseline_stats_path.exists():  # Prefer persisted feature order when available.
            with open(baseline_stats_path, "r", encoding="utf-8") as f:  # Read the baseline stats artifact.
                baseline_stats = json.load(f)

            cols = baseline_stats.get("feature_columns", [])  # Extract stored feature column order.
            if isinstance(cols, list) and cols:  # Use stored order only when it looks valid.
                return [str(v) for v in cols]
        return list(FEATURE_MAPPING.values())  # Fallback to hard-coded default feature order.

    def load_model(self) -> None:
        """Load the trained model from disk and update the model-loaded metric."""

        model_path = Path(self.paths["model_file"])  # Location of the deployed model artifact.
        if not model_path.exists():  # API can start even when no model has been trained yet.
            self.model = None
            MODEL_LOADED.set(0)
            return

        self.model = joblib.load(model_path)  # Load model into memory for low-latency inference.
        MODEL_LOADED.set(1)  # Update Prometheus gauge to show readiness.

    def append_live_row(self, row: Dict[str, float | int]) -> None:
        """Append a prediction request and its predicted class to the live-data log."""

        live_data_path = Path(self.paths["live_data"])  # CSV used later for drift checks and retraining.
        ensure_parent_dir(str(live_data_path))  # Create containing folder if needed.

        should_write_header = not live_data_path.exists()  # Write CSV header only for a new file.
        with open(live_data_path, "a", newline="", encoding="utf-8") as f:  # Append a single row to the live log.
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if should_write_header:
                writer.writeheader()  # Add column names when the file is created.
            writer.writerow(row)  # Persist the new live prediction row.

    def drift_status_path(self) -> Path:
        """Return the configured drift-status file path."""

        return Path(self.paths.get("drift_status", "drift/drift_status.json"))


def _drift_status_to_value(status: str) -> float:
    """Convert drift status text into a numeric Prometheus-friendly value."""

    if status == "ok":  # Healthy drift state.
        return 0.0
    if status == "warning":  # Drift is elevated but not critical.
        return 1.0
    if status == "alert":  # Drift crossed the alert threshold.
        return 2.0
    return -1.0  # Missing or insufficient drift signal.


def _load_drift_status() -> Dict[str, object]:
    """Load the latest drift result from disk, or return a default status."""

    status_path = service.drift_status_path()  # Drift monitor persists the latest result here.
    if not status_path.exists():  # Handle the case where no drift check has run yet.
        return {
            "status": "insufficient_data",
            "message": "Drift check has not run yet.",
            "features": {},
            "live_rows": 0,
        }

    with open(status_path, "r", encoding="utf-8") as f:  # Read saved drift status JSON.
        return json.load(f)


def _update_drift_gauges(drift_status: Dict[str, object]) -> None:
    """Push the latest drift results into Prometheus gauges."""

    status_text = str(drift_status.get("status", "insufficient_data"))  # Read current overall status.
    DRIFT_STATUS.set(_drift_status_to_value(status_text))  # Publish overall numeric status.

    features = drift_status.get("features", {})  # Per-feature drift results from the last check.
    max_ks = 0.0  # Track the largest observed KS score.
    if isinstance(features, dict):
        for feature_name, feature_result in features.items():  # Publish each feature's drift score.
            ks_value = 0.0
            if isinstance(feature_result, dict):
                ks_value = float(feature_result.get("ks_stat", 0.0))
            DRIFT_FEATURE_KS.labels(feature=str(feature_name)).set(ks_value)
            if ks_value > max_ks:
                max_ks = ks_value
    DRIFT_MAX_KS.set(max_ks)  # Publish the largest per-feature KS score.


def _predict_class_probabilities(model: object, feature_df: pd.DataFrame) -> Dict[int, float]:
    """Return per-class probabilities when the model exposes predict_proba."""

    if not hasattr(model, "predict_proba"):
        return {}

    try:
        probabilities = model.predict_proba(feature_df)[0]
    except Exception:
        return {}

    raw_classes = getattr(model, "classes_", list(range(len(probabilities))))
    class_probabilities: Dict[int, float] = {}
    for class_id, probability in zip(raw_classes, probabilities):
        try:
            normalized_class_id = int(class_id)
        except (TypeError, ValueError):
            continue
        class_probabilities[normalized_class_id] = float(probability)
    return class_probabilities


service = ModelService()  # Shared service object used by all API routes.
service.load_model()  # Load the trained model once at startup.

app = FastAPI(title="Simple Iris Prediction API")  # Main FastAPI application instance.


@app.get("/health")
def health() -> Dict[str, object]:
    """Simple readiness endpoint for monitoring and quick checks."""

    return {
        "status": "ok",
        "model_loaded": bool(service.model is not None),
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> Dict[str, object]:
    """Run inference for a single request and log the live feature row."""

    start = perf_counter()  # Start latency timer as early as possible.
    latency_seconds: float | None = None

    try:
        if service.model is None:  # Reject requests when no trained model is available.
            PREDICTION_REQUESTS.labels(endpoint="predict", status="no_model").inc()
            raise HTTPException(status_code=503, detail="Model is not available. Train first.")

        feature_row = {
            FEATURE_MAPPING["sepal_length_cm"]: payload.sepal_length_cm,  # Map API field to model feature name.
            FEATURE_MAPPING["sepal_width_cm"]: payload.sepal_width_cm,  # Map API field to model feature name.
            FEATURE_MAPPING["petal_length_cm"]: payload.petal_length_cm,  # Map API field to model feature name.
            FEATURE_MAPPING["petal_width_cm"]: payload.petal_width_cm,  # Map API field to model feature name.
        }
        feature_df = pd.DataFrame([feature_row])[service.feature_columns]  # Preserve trained feature ordering for inference.

        prediction = int(service.model.predict(feature_df)[0])  # Run the model and extract the first prediction.
        class_name = CLASS_MAPPING.get(prediction, "unknown")  # Convert numeric class to readable label.
        class_probabilities = _predict_class_probabilities(service.model, feature_df)  # Collect class probabilities when supported.

        if class_probabilities:
            confidence_score = float(class_probabilities.get(prediction, max(class_probabilities.values())))
        else:
            confidence_score = 1.0

        live_row = {
            **feature_row,
            "predicted_class": prediction,
        }
        service.append_live_row(live_row)  # Save request and prediction for drift monitoring and retraining.

        PREDICTION_CLASS_TOTAL.labels(class_id=str(prediction)).inc()  # Count predicted class.
        PREDICTION_REQUESTS.labels(endpoint="predict", status="ok").inc()  # Count successful requests.

        latency_seconds = perf_counter() - start  # Compute request latency once so we can return and emit the same value.
        class_probability_by_label = {
            CLASS_MAPPING.get(class_id, str(class_id)): probability
            for class_id, probability in sorted(class_probabilities.items())
        }

        return {
            "predicted_class": prediction,
            "predicted_label": class_name,
            "confidence": confidence_score,
            "class_probabilities": class_probability_by_label,
            "latency_ms": round(latency_seconds * 1000.0, 3),
        }
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_REQUESTS.labels(endpoint="predict", status="error").inc()  # Count failed prediction attempts.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        if latency_seconds is None:
            latency_seconds = perf_counter() - start
        PREDICTION_LATENCY_SECONDS.observe(latency_seconds)  # Record request latency whether success or failure.


@app.post("/drift/check")
def drift_check() -> Dict[str, object]:
    """Trigger a fresh drift check and return the latest result."""

    run_drift_check()  # Execute the batch drift-monitor logic.
    status = _load_drift_status()  # Read the newly written drift result.
    _update_drift_gauges(status)  # Publish latest drift metrics to Prometheus.
    return status


@app.get("/drift/status")
def drift_status() -> Dict[str, object]:
    """Return the most recent drift result without re-running the check."""

    status = _load_drift_status()  # Read most recent persisted drift result.
    _update_drift_gauges(status)  # Keep Prometheus gauges in sync with returned status.
    return status


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in text format."""

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":  # Allow local execution with python -m src.serve_api.
    params = load_params()  # Read host and port from config.
    host = params["service"]["host"]
    port = int(params["service"]["port"])
    uvicorn.run(app, host=host, port=port, reload=False)  # Start production-style single-process server.
