from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("IRIS_API_URL", "http://127.0.0.1:8000")
LIVE_REQUESTS_PATH = Path("drift/live_requests.csv")


def api_get_json(base_url: str, endpoint: str, timeout: int = 10) -> Tuple[bool, Dict[str, Any] | str]:
    try:
        response = requests.get(f"{base_url}{endpoint}", timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def api_post_json(
    base_url: str,
    endpoint: str,
    payload: Dict[str, Any] | None = None,
    timeout: int = 10,
) -> Tuple[bool, Dict[str, Any] | str]:
    try:
        response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def status_label(status: str) -> str:
    mapping = {
        "ok": "OK",
        "warning": "WARNING",
        "alert": "ALERT",
        "insufficient_data": "INSUFFICIENT DATA",
    }
    return mapping.get(status, status.upper())


st.set_page_config(page_title="Iris Predictor + Drift", layout="wide")

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

st.title("Simple Iris Live Prediction")
st.caption("Real-time inference + drift visibility using FastAPI, Prometheus, and Streamlit.")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL)
    st.caption("Example: http://127.0.0.1:8000")

ok_health, health_result = api_get_json(api_url, "/health")
if ok_health:
    st.success("API is reachable.")
else:
    st.error(f"API connection failed: {health_result}")

left_col, right_col = st.columns([1.3, 1.0], gap="large")

with left_col:
    st.subheader("Predict")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            sepal_length_cm = st.number_input("Sepal length (cm)", min_value=0.0, value=5.1, step=0.1)
            petal_length_cm = st.number_input("Petal length (cm)", min_value=0.0, value=1.4, step=0.1)
        with c2:
            sepal_width_cm = st.number_input("Sepal width (cm)", min_value=0.0, value=3.5, step=0.1)
            petal_width_cm = st.number_input("Petal width (cm)", min_value=0.0, value=0.2, step=0.1)

        predict_clicked = st.form_submit_button("Get Prediction", use_container_width=True)

    if predict_clicked:
        payload = {
            "sepal_length_cm": sepal_length_cm,
            "sepal_width_cm": sepal_width_cm,
            "petal_length_cm": petal_length_cm,
            "petal_width_cm": petal_width_cm,
        }
        ok_pred, pred_result = api_post_json(api_url, "/predict", payload)
        if ok_pred:
            st.metric("Predicted Label", str(pred_result.get("predicted_label", "unknown")).title())
            st.metric("Predicted Class ID", int(pred_result.get("predicted_class", -1)))
        else:
            st.error(f"Prediction failed: {pred_result}")

    st.subheader("Recent Requests")
    if LIVE_REQUESTS_PATH.exists():
        df_live = pd.read_csv(LIVE_REQUESTS_PATH)
        st.dataframe(df_live.tail(10), use_container_width=True, hide_index=True)
    else:
        st.info("No live prediction rows yet.")

with right_col:
    st.subheader("Drift Status")
    refresh_col, check_col = st.columns(2)

    with refresh_col:
        refresh_status = st.button("Refresh Status", use_container_width=True)
    with check_col:
        run_check = st.button("Run Drift Check", use_container_width=True)

    endpoint = "/drift/check" if run_check else "/drift/status"
    method_ok, drift_result = (
        api_post_json(api_url, endpoint) if run_check else api_get_json(api_url, endpoint)
    )

    if method_ok and isinstance(drift_result, dict):
        status = str(drift_result.get("status", "unknown"))
        st.metric("Overall Drift", status_label(status))
        st.metric("Live Rows", int(drift_result.get("live_rows", 0)))
        st.metric("Window Size", int(drift_result.get("window_size", 0)))
        st.caption(str(drift_result.get("message", "")))

        feature_map = drift_result.get("features", {})
        if isinstance(feature_map, dict) and feature_map:
            rows = []
            for feature_name, result in feature_map.items():
                if not isinstance(result, dict):
                    continue
                rows.append(
                    {
                        "feature": feature_name,
                        "ks_stat": float(result.get("ks_stat", 0.0)),
                        "p_value": float(result.get("p_value", 0.0)),
                        "status": str(result.get("status", "unknown")),
                    }
                )
            if rows:
                drift_df = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
                st.dataframe(drift_df, use_container_width=True, hide_index=True)
        else:
            st.info("No per-feature drift stats yet.")
    else:
        st.error(f"Failed to fetch drift status: {drift_result}")

if refresh_status:
    st.rerun()
