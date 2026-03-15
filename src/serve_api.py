from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from src.common import ensure_parent_dir, load_params
from src.drift_monitor import run_drift_check

FEATURE_MAPPING: Dict[str, str] = {
    "sepal_length_cm": "sepal length (cm)",
    "sepal_width_cm": "sepal width (cm)",
    "petal_length_cm": "petal length (cm)",
    "petal_width_cm": "petal width (cm)",
}

CLASS_MAPPING: Dict[int, str] = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests.",
    ["endpoint", "status"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds.",
)

PREDICTION_CLASS_TOTAL = Counter(
    "prediction_class_total",
    "Count of predictions by class.",
    ["class_id"],
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether a model artifact is loaded (1) or not (0).",
)

DRIFT_STATUS = Gauge(
    "data_drift_status",
    "Current drift state as numeric value (ok=0, warning=1, alert=2, insufficient_data=-1).",
)

DRIFT_MAX_KS = Gauge(
    "data_drift_max_ks_stat",
    "Maximum KS drift statistic across monitored features from latest check.",
)

DRIFT_FEATURE_KS = Gauge(
    "data_drift_feature_ks_stat",
    "KS drift statistic for each monitored feature from latest check.",
    ["feature"],
)

# The PredictRequest class defines the expected structure of 
# the input data for making predictions. It includes fields for 
# sepal length, sepal width, petal length, and petal width, all of
# which are required to be floating-point numbers. This class is 
# used by FastAPI to validate incoming JSON payloads for the /predict 
# endpoint, ensuring that the necessary features are provided in the correct 
# format before making a prediction with the trained model.
class PredictRequest(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

# The ModelService class encapsulates the logic for loading the trained model,
# managing feature columns, and appending live prediction data to a CSV file.
class ModelService:
    def __init__(self) -> None:
        self.params = load_params()
        self.paths = self.params["paths"]
        self.model = None
        self.feature_columns = self._load_feature_columns()

    # The _load_feature_columns method attempts to load the feature columns
    # from the baseline statistics file. If the file exists and contains a 
    # valid list of feature columns, it returns that list. Otherwise, it 
    # falls back to using the default feature names defined in the 
    # FEATURE_MAPPING dictionary. This allows the service to maintain 
    # consistency in feature ordering and naming, which is crucial for 
    # making accurate predictions with the trained model.
    def _load_feature_columns(self) -> List[str]:
        baseline_stats_path = Path(self.paths["baseline_stats"])
        if baseline_stats_path.exists():
            with open(baseline_stats_path, "r", encoding="utf-8") as f:
                baseline_stats = json.load(f)

            cols = baseline_stats.get("feature_columns", [])
            if isinstance(cols, list) and cols:
                return [str(v) for v in cols]
        return list(FEATURE_MAPPING.values())

    def load_model(self) -> None:
        model_path = Path(self.paths["model_file"])
        if not model_path.exists():
            self.model = None
            MODEL_LOADED.set(0)
            return

        self.model = joblib.load(model_path)
        MODEL_LOADED.set(1)

    def append_live_row(self, row: Dict[str, float | int]) -> None:
        live_data_path = Path(self.paths["live_data"])
        ensure_parent_dir(str(live_data_path))

        should_write_header = not live_data_path.exists()
        with open(live_data_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if should_write_header:
                writer.writeheader()
            writer.writerow(row)

    def drift_status_path(self) -> Path:
        return Path(self.paths.get("drift_status", "drift/drift_status.json"))


def _drift_status_to_value(status: str) -> float:
    if status == "ok":
        return 0.0
    if status == "warning":
        return 1.0
    if status == "alert":
        return 2.0
    return -1.0


def _load_drift_status() -> Dict[str, object]:
    status_path = service.drift_status_path()
    if not status_path.exists():
        return {
            "status": "insufficient_data",
            "message": "Drift check has not run yet.",
            "features": {},
            "live_rows": 0,
        }

    with open(status_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _update_drift_gauges(drift_status: Dict[str, object]) -> None:
    status_text = str(drift_status.get("status", "insufficient_data"))
    DRIFT_STATUS.set(_drift_status_to_value(status_text))

    features = drift_status.get("features", {})
    max_ks = 0.0
    if isinstance(features, dict):
        for feature_name, feature_result in features.items():
            ks_value = 0.0
            if isinstance(feature_result, dict):
                ks_value = float(feature_result.get("ks_stat", 0.0))
            DRIFT_FEATURE_KS.labels(feature=str(feature_name)).set(ks_value)
            if ks_value > max_ks:
                max_ks = ks_value
    DRIFT_MAX_KS.set(max_ks)


service = ModelService()
service.load_model()

app = FastAPI(title="Simple Iris Prediction API")


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": bool(service.model is not None),
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> Dict[str, object]:
    start = perf_counter()

    if service.model is None:
        PREDICTION_REQUESTS.labels(endpoint="predict", status="no_model").inc()
        raise HTTPException(status_code=503, detail="Model is not available. Train first.")

    try:
        feature_row = {
            FEATURE_MAPPING["sepal_length_cm"]: payload.sepal_length_cm,
            FEATURE_MAPPING["sepal_width_cm"]: payload.sepal_width_cm,
            FEATURE_MAPPING["petal_length_cm"]: payload.petal_length_cm,
            FEATURE_MAPPING["petal_width_cm"]: payload.petal_width_cm,
        }
        feature_df = pd.DataFrame([feature_row])[service.feature_columns]

        prediction = int(service.model.predict(feature_df)[0])
        class_name = CLASS_MAPPING.get(prediction, "unknown")

        live_row = {
            **feature_row,
            "predicted_class": prediction,
        }
        service.append_live_row(live_row)

        PREDICTION_CLASS_TOTAL.labels(class_id=str(prediction)).inc()
        PREDICTION_REQUESTS.labels(endpoint="predict", status="ok").inc()

        return {
            "predicted_class": prediction,
            "predicted_label": class_name,
        }
    except Exception as exc:
        PREDICTION_REQUESTS.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        PREDICTION_LATENCY_SECONDS.observe(perf_counter() - start)


@app.post("/drift/check")
def drift_check() -> Dict[str, object]:
    run_drift_check()
    status = _load_drift_status()
    _update_drift_gauges(status)
    return status


@app.get("/drift/status")
def drift_status() -> Dict[str, object]:
    status = _load_drift_status()
    _update_drift_gauges(status)
    return status


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    params = load_params()
    host = params["service"]["host"]
    port = int(params["service"]["port"])
    uvicorn.run(app, host=host, port=port, reload=False)
