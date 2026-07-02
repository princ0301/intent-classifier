from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def get_or_create_experiment(name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return mlflow.create_experiment(name)
    return experiment.experiment_id


def log_config(config: dict) -> None:
    flat = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[f"{section}.{k}"] = v
        else:
            flat[section] = values
    mlflow.log_params(flat)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    save_path: str = "artifacts/confusion_matrix.png",
) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

    mlflow.log_artifact(str(path))


def register_model(run_id: str, artifact_path: str, model_name: str, alias: str = "champion") -> None:
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, model_name)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=result.version,
    )
    print(f"  registered {model_name} v{result.version} with alias '{alias}'")


def load_registered_model(model_name: str, alias: str = "champion"):
    model_uri = f"models:/{model_name}@{alias}"
    return mlflow.pyfunc.load_model(model_uri)


def get_model_version_by_alias(model_name: str, alias: str = "champion"):
    client = mlflow.MlflowClient()
    return client.get_model_version_by_alias(model_name, alias)
