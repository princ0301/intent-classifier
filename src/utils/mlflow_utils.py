import numpy as np
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
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

def register_model(run_id: str, model_name: str, stage: str = "Staging") -> None:
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage=stage,
    )
 
def load_registered_model(model_name: str, stage: str = "Production"):
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)