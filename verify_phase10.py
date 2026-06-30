import mlflow

from src.utils.mlflow_utils import get_model_version_by_alias, load_registered_model
from src.utils.settings import settings


def main():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    print("checking registered models...")
    for model_type, alias in [("logreg", "challenger"), ("svm", "challenger"), ("distilbert", "champion")]:
        name = f"intent-classifier-{model_type}"
        version = get_model_version_by_alias(name, alias)
        print(f"  {name}@{alias} -> version {version.version}, run_id={version.run_id}")

    print("\nloading champion model (distilbert) via alias...")
    model = load_registered_model("intent-classifier-distilbert", alias="champion")
    print(f"  loaded: {type(model)}")


if __name__ == "__main__":
    main()