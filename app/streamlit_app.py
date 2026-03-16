from __future__ import annotations  # Allow postponed evaluation of type annotations.

import os  # Read optional environment overrides.
from pathlib import Path  # Handle filesystem paths safely.
from typing import Any, Dict, Tuple  # Type hints for API helper responses.

import pandas as pd  # Build tables and charts from API results.
import requests  # Call the FastAPI service from the Streamlit frontend.
import streamlit as st  # Render the interactive web UI.

DEFAULT_API_URL = os.getenv("IRIS_API_URL", "http://127.0.0.1:8000")  # Default backend URL, overridable via environment variable.
LIVE_REQUESTS_PATH = Path("drift/live_requests.csv")  # Local CSV where the API appends recent prediction requests.


def api_get_json(base_url: str, endpoint: str, timeout: int = 10) -> Tuple[bool, Dict[str, Any] | str]:
    """Call a GET endpoint and return either parsed JSON or an error string."""

    try:
        response = requests.get(f"{base_url}{endpoint}", timeout=timeout)  # Send GET request to the backend service.
        response.raise_for_status()  # Raise an exception for non-2xx responses.
        return True, response.json()  # Return parsed JSON on success.
    except Exception as exc:
        return False, str(exc)  # Return error text so the UI can display it.


def api_post_json(
    base_url: str,
    endpoint: str,
    payload: Dict[str, Any] | None = None,
    timeout: int = 10,
) -> Tuple[bool, Dict[str, Any] | str]:
    """Call a POST endpoint with JSON payload and return parsed JSON or an error."""

    try:
        response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=timeout)  # Send POST request to the backend service.
        response.raise_for_status()  # Raise an exception for non-2xx responses.
        return True, response.json()  # Return parsed JSON on success.
    except Exception as exc:
        return False, str(exc)  # Return error text so the UI can display it.


def status_label(status: str) -> str:
    """Convert backend drift-status codes into user-friendly labels."""

    mapping = {
        "ok": "OK",  # No drift detected.
        "warning": "WARNING",  # Drift is elevated but not critical.
        "alert": "ALERT",  # Drift crossed the alert threshold.
        "insufficient_data": "INSUFFICIENT DATA",  # Not enough live rows to evaluate drift.
    }
    return mapping.get(status, status.upper())  # Fall back to uppercase for unknown statuses.


st.set_page_config(page_title="Iris Predictor + Drift", layout="wide")  # Configure page title and wide layout.

# Inject a small custom theme to make the dashboard more readable and visually polished.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #d9f0ff 0%, #f8fbff 45%, #edf6f2 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #d5e4ef;
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Simple Iris Live Prediction")  # Main page title.
st.caption("Real-time inference + drift visibility using FastAPI, Prometheus, and Streamlit.")  # Short summary of what the UI shows.

# Sidebar holds connection settings so the main layout stays focused on predictions and monitoring.
with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL)  # Allow user to point the UI at a different backend.
    st.caption("Example: http://127.0.0.1:8000")

# Check whether the backend is reachable before rendering interactive actions.
ok_health, health_result = api_get_json(api_url, "/health")
if ok_health:
    st.success("API is reachable.")
else:
    st.error(f"API connection failed: {health_result}")

# Split the page into prediction and monitoring areas.
left_col, right_col = st.columns([1.3, 1.0], gap="large")

with left_col:
    st.subheader("Predict")

    # Prediction inputs are wrapped in a form so the request is sent only on submit.
    with st.form("predict_form"):
        c1, c2 = st.columns(2)  # Arrange feature inputs into two columns.
        with c1:
            sepal_length_cm = st.number_input("Sepal length (cm)", min_value=0.0, value=5.1, step=0.1)  # Default sample value from Iris setosa.
            petal_length_cm = st.number_input("Petal length (cm)", min_value=0.0, value=1.4, step=0.1)  # Default sample value from Iris setosa.
        with c2:
            sepal_width_cm = st.number_input("Sepal width (cm)", min_value=0.0, value=3.5, step=0.1)  # Default sample value from Iris setosa.
            petal_width_cm = st.number_input("Petal width (cm)", min_value=0.0, value=0.2, step=0.1)  # Default sample value from Iris setosa.

        predict_clicked = st.form_submit_button("Get Prediction", use_container_width=True)  # Submit button for a single prediction.

    if predict_clicked:
        payload = {
            "sepal_length_cm": sepal_length_cm,  # User-entered sepal length.
            "sepal_width_cm": sepal_width_cm,  # User-entered sepal width.
            "petal_length_cm": petal_length_cm,  # User-entered petal length.
            "petal_width_cm": petal_width_cm,  # User-entered petal width.
        }
        ok_pred, pred_result = api_post_json(api_url, "/predict", payload)  # Ask the backend to score the entered features.
        if ok_pred and isinstance(pred_result, dict):
            metric_cols = st.columns(4)  # Show prediction summary across four compact tiles.
            with metric_cols[0]:
                st.metric("Predicted Label", str(pred_result.get("predicted_label", "unknown")).title())  # Human-readable class name.
            with metric_cols[1]:
                st.metric("Predicted Class ID", int(pred_result.get("predicted_class", -1)))  # Numeric class ID returned by the backend.
            with metric_cols[2]:
                confidence_value = pred_result.get("confidence")  # Optional confidence field if the API exposes it.
                if isinstance(confidence_value, (int, float)):
                    st.metric("Confidence", f"{float(confidence_value) * 100.0:.2f}%")
                else:
                    st.metric("Confidence", "N/A")  # Gracefully handle APIs that do not return confidence.
            with metric_cols[3]:
                latency_ms = pred_result.get("latency_ms")  # Optional latency field if the API exposes it.
                if isinstance(latency_ms, (int, float)):
                    st.metric("Latency (ms)", f"{float(latency_ms):.2f}")
                else:
                    st.metric("Latency (ms)", "N/A")  # Gracefully handle APIs that do not return latency.

            class_probabilities = pred_result.get("class_probabilities", {})  # Optional per-class probabilities from the backend.
            if isinstance(class_probabilities, dict) and class_probabilities:
                probability_rows = []  # Collect chart rows only for valid numeric probability values.
                for class_name, probability in class_probabilities.items():
                    try:
                        probability_value = float(probability)  # Normalize probability to float for plotting.
                    except (TypeError, ValueError):
                        continue  # Skip invalid entries instead of failing the whole UI block.
                    probability_rows.append(
                        {
                            "class": str(class_name).title(),
                            "probability": probability_value,
                        }
                    )

                if probability_rows:
                    probability_df = pd.DataFrame(probability_rows).sort_values("probability", ascending=False)  # Highest-probability classes shown first.
                    st.caption("Class probability breakdown")
                    st.bar_chart(probability_df.set_index("class"))  # Visualize per-class probabilities.
        else:
            st.error(f"Prediction failed: {pred_result}")  # Show backend error to the user.

    st.subheader("Recent Requests")
    if LIVE_REQUESTS_PATH.exists():
        df_live = pd.read_csv(LIVE_REQUESTS_PATH)  # Read the local live-request log produced by the API.
        st.dataframe(df_live.tail(10), use_container_width=True, hide_index=True)  # Show the most recent rows only.
    else:
        st.info("No live prediction rows yet.")

with right_col:
    st.subheader("Drift Status")
    refresh_col, check_col = st.columns(2)  # Separate refresh from active drift-check trigger.

    with refresh_col:
        refresh_status = st.button("Refresh Status", use_container_width=True)  # Re-read latest stored drift result.
    with check_col:
        run_check = st.button("Run Drift Check", use_container_width=True)  # Trigger a fresh drift computation via the API.

    endpoint = "/drift/check" if run_check else "/drift/status"  # Use POST to trigger a check, otherwise GET the latest saved status.
    method_ok, drift_result = (
        api_post_json(api_url, endpoint) if run_check else api_get_json(api_url, endpoint)
    )

    if method_ok and isinstance(drift_result, dict):
        status = str(drift_result.get("status", "unknown"))  # Read overall drift severity.
        st.metric("Overall Drift", status_label(status))  # Display normalized status label.
        st.metric("Live Rows", int(drift_result.get("live_rows", 0)))  # Show how many live rows are currently available.
        st.metric("Window Size", int(drift_result.get("window_size", 0)))  # Show how many recent rows were compared.
        st.caption(str(drift_result.get("message", "")))  # Display backend explanation or summary text.

        feature_map = drift_result.get("features", {})  # Per-feature KS statistics.
        if isinstance(feature_map, dict) and feature_map:
            rows = []  # Build a flat table for Streamlit display.
            for feature_name, result in feature_map.items():
                if not isinstance(result, dict):
                    continue  # Skip malformed feature entries.
                rows.append(
                    {
                        "feature": feature_name,
                        "ks_stat": float(result.get("ks_stat", 0.0)),
                        "p_value": float(result.get("p_value", 0.0)),
                        "status": str(result.get("status", "unknown")),
                    }
                )
            if rows:
                drift_df = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)  # Show most-drifted features first.
                st.dataframe(drift_df, use_container_width=True, hide_index=True)
        else:
            st.info("No per-feature drift stats yet.")
    else:
        st.error(f"Failed to fetch drift status: {drift_result}")  # Show backend or network error.

# Force Streamlit to rerun the script so it re-fetches current API/drift data.
if refresh_status:
    st.rerun()
