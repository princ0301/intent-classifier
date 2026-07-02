import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def compute_latency(predict_fn, inputs, n_runs: int = 100) -> dict:
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(inputs)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return {
        "latency_mean_ms": round(float(np.mean(latencies)), 3),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 3),
    }


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> str:
    return classification_report(y_true, y_pred, target_names=label_names, zero_division=0)


def compare_models(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        rows.append({"model": model_name, **metrics})
    return pd.DataFrame(rows).set_index("model")
