# Prometheus Query Guide

This file contains small PromQL queries you can paste into the Prometheus UI at `http://127.0.0.1:9090`.

## 1. Model Availability

```promql
model_loaded
```

Interpretation:
- `1` means the API has a model loaded.
- `0` means the API is running without a model.

## 2. Prediction Request Rate

```promql
sum(rate(prediction_requests_total{endpoint="predict",status="ok"}[5m]))
```

Interpretation:
- Shows successful prediction throughput over the last 5 minutes.

## 3. Prediction Error Rate

```promql
sum(rate(prediction_requests_total{endpoint="predict",status="error"}[5m]))
```

Interpretation:
- Shows how often prediction requests are failing.

## 4. Prediction Latency p95

```promql
histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))
```

Interpretation:
- Estimates the 95th percentile prediction latency in seconds.

## 5. Predicted Class Distribution

```promql
sum by (class_id) (rate(prediction_class_total[5m]))
```

Interpretation:
- Shows which classes the model is predicting most often.

## 6. Overall Drift Status

```promql
data_drift_status
```

Interpretation:
- `-1` = insufficient data
- `0` = ok
- `1` = warning
- `2` = alert

## 7. Maximum Drift Score

```promql
data_drift_max_ks_stat
```

Interpretation:
- Shows the highest KS statistic across monitored features from the latest drift check.

## 8. Per-Feature Drift Score

```promql
data_drift_feature_ks_stat
```

Interpretation:
- Shows KS drift score for each feature label.

## 9. Active Alerts

```promql
ALERTS{alertstate="firing"}
```

Interpretation:
- Lists currently firing Prometheus alerts.

## Suggested First Dashboard

If you make a simple dashboard later, use these first panels:
- single stat: `model_loaded`
- single stat: `data_drift_status`
- time series: `data_drift_max_ks_stat`
- time series: `sum(rate(prediction_requests_total{endpoint="predict",status="ok"}[5m]))`
- time series: `histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))`
- table or time series: `data_drift_feature_ks_stat`