import mlflow
import mlflow.sklearn
import mlflow.transformers

# from src.data.preprocessor import load_label_map
from src.features.tfidf import load_vectorizer, transform
from src.models.classical import LogisticRegressionModel, SVMModel
from src.models.transformer import TransformerModel
from src.utils.mlflow_utils import get_or_create_experiment, register_model
from src.utils.settings import settings

LOGREG_PATH = "artifacts/models/logreg.pkl"
SVM_PATH = "artifacts/models/svm.pkl"
DISTILBERT_DIR = "artifacts/models/distilbert"
VECTORIZER_PATH = "artifacts/vectorizers/tfidf.pkl"

REGISTRY_NAME = "intent-classifier"


def register_sklearn_model(model_obj, model_type: str, sample_input, experiment_id: str) -> None:
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"register-{model_type}") as run:
        mlflow.sklearn.log_model(
            sk_model=model_obj.model,
            artifact_path="model",
            input_example=sample_input,
        )
        mlflow.set_tag("model_type", model_type)
        register_model(
            run_id=run.info.run_id,
            artifact_path="model",
            model_name=f"{REGISTRY_NAME}-{model_type}",
            alias="challenger",
        )


def register_transformer_model(experiment_id: str) -> None:
    transformer = TransformerModel(model_name=DISTILBERT_DIR, num_labels=151)

    with mlflow.start_run(experiment_id=experiment_id, run_name="register-distilbert") as run:
        mlflow.transformers.log_model(
            transformers_model={
                "model": transformer.model,
                "tokenizer": transformer.tokenizer,
            },
            artifact_path="model",
            task="text-classification",
        )
        mlflow.set_tag("model_type", "distilbert")
        register_model(
            run_id=run.info.run_id,
            artifact_path="model",
            model_name=f"{REGISTRY_NAME}-distilbert",
            alias="champion",
        )


def main():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_id = get_or_create_experiment("intent-classifier")

    # label_map = load_label_map()

    print("registering logistic regression...")
    vectorizer = load_vectorizer(VECTORIZER_PATH)
    logreg = LogisticRegressionModel()
    logreg.load(LOGREG_PATH)
    sample = transform(vectorizer, ["what is my balance"])
    register_sklearn_model(logreg, "logreg", sample, experiment_id)

    print("registering svm...")
    svm = SVMModel()
    svm.load(SVM_PATH)
    register_sklearn_model(svm, "svm", sample, experiment_id)

    print("registering distilbert as champion...")
    register_transformer_model(experiment_id)

    print("\nregistry summary:")
    client = mlflow.MlflowClient()
    for name in [
        f"{REGISTRY_NAME}-logreg",
        f"{REGISTRY_NAME}-svm",
        f"{REGISTRY_NAME}-distilbert",
    ]:
        versions = client.search_model_versions(f"name='{name}'")
        for v in versions:
            aliases = v.aliases if hasattr(v, "aliases") else []
            print(f"  {name} v{v.version} aliases={aliases}")


if __name__ == "__main__":
    main()
