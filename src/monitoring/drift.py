from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE_PATH = "artifacts/monitoring/reference_data.csv"
CURRENT_PATH = "artifacts/monitoring/current_data.csv"


def save_reference_data(
    texts: list[str],
    predictions: list[str],
    confidences: list[float],
    save_path: str = REFERENCE_PATH,
) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "text": texts,
            "text_length": [len(t) for t in texts],
            "prediction": predictions,
            "confidence": confidences,
        }
    )
    df.to_csv(path, index=False)


def append_current_data(
    texts: list[str],
    predictions: list[str],
    confidences: list[float],
    save_path: str = CURRENT_PATH,
    window_size: int = 1000,
) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(
        {
            "text": texts,
            "text_length": [len(t) for t in texts],
            "prediction": predictions,
            "confidence": confidences,
        }
    )

    if path.exists():
        existing_df = pd.read_csv(path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.tail(window_size)
    else:
        combined = new_df

    combined.to_csv(path, index=False)


def load_reference_data(load_path: str = REFERENCE_PATH) -> pd.DataFrame:
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference data not found: {path}")
    return pd.read_csv(path)


def load_current_data(load_path: str = CURRENT_PATH) -> pd.DataFrame:
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Current data not found: {path}")
    return pd.read_csv(path)


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Report:
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df, current_data=current_df)
    return snapshot


def get_drift_summary(snapshot) -> dict:
    result_dict = snapshot.dict()

    summary = {
        "drifted_columns": [],
        "total_columns": 0,
        "drift_share": 0.0,
    }

    for metric in result_dict.get("metrics", []):
        metric_id = metric.get("metric_id", "")
        value = metric.get("value", {})

        if "DriftedColumnsCount" in metric_id:
            summary["drifted_columns_count"] = value.get("count", 0)
            summary["drift_share"] = value.get("share", 0.0)

        if "ValueDrift" in metric_id and isinstance(value, (int, float)):
            column_name = metric.get("metric_name", metric_id)
            if value > 0.5:
                summary["drifted_columns"].append(
                    {
                        "column": column_name,
                        "drift_score": round(value, 4),
                    }
                )

    return summary


def save_drift_report_html(snapshot, save_path: str = "artifacts/monitoring/drift_report.html") -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))


def get_confidence_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    ref_mean = reference_df["confidence"].mean()
    cur_mean = current_df["confidence"].mean()
    drop = ref_mean - cur_mean

    return {
        "reference_avg_confidence": round(float(ref_mean), 4),
        "current_avg_confidence": round(float(cur_mean), 4),
        "confidence_drop": round(float(drop), 4),
        "is_degraded": bool(drop > 0.1),
    }


def get_oos_rate_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    oos_threshold: float = 0.5,
) -> dict:
    ref_oos_rate = (reference_df["confidence"] < oos_threshold).mean()
    cur_oos_rate = (current_df["confidence"] < oos_threshold).mean()
    increase = cur_oos_rate - ref_oos_rate

    return {
        "reference_oos_rate": round(float(ref_oos_rate), 4),
        "current_oos_rate": round(float(cur_oos_rate), 4),
        "oos_rate_increase": round(float(increase), 4),
        "is_anomalous": bool(increase > 0.15),
    }
