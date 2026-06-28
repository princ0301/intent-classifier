import os
import mlflow
import numpy as np
from src.data.loader import load_clinc150, load_splits, save_splits
from src.data.preprocessor import load_label_map, preprocess
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_latency,
    get_classification_report,
)
from src.features.tfidf import fit_tfidf, load_vectorizer, save_vectorizer, transform
from src.models.classical import LogisticRegressionModel, SVMModel
from src.storage.s3 import upload_artifact
from src.utils.config import load_config
from src.utils.mlflow_utils import (
    get_or_create_experiment,
    log_config,
    log_confusion_matrix,
    log_metrics,
)
from src.utils.settings import settings

DATA_DIR = "data/raw"
LOGREG_PATH = "artifacts/models/logreg.pkl"
SVM_PATH = "artifacts/models/svm.pkl"
VECTORIZER_PATH = "artifacts/vectorizers/tfidf.pkl"

def get_config_value(config: dict, key: str):
    kebab_key = key.replace("_", "-")
    if key in config:
        return config[key]
    if kebab_key in config:
        return config[kebab_key]
    raise KeyError(key)

def load_or_download_data(config: dict) -> tuple:
    train_path = os.path.join(DATA_DIR, "train.csv")
    if os.path.exists(train_path):
        print("loading data from disk...")
        splits = load_splits(DATA_DIR)
    else:
        print("downloading CLINC150...")
        splits = load_clinc150(config["data"]["subset"])
        save_splits(splits, DATA_DIR)

    processed, label_map = preprocess(splits)
    return processed, label_map

def get_features(processed: dict, config: dict) -> tuple:
    train_texts = processed["train"]["text"].tolist()
    val_texts = processed["validation"]["text"].tolist()
    test_texts = processed["test"]["text"].tolist()

    try:
        print("loading existing vectorizer...")
        vectorizer = load_vectorizer(VECTORIZER_PATH)
    except FileNotFoundError:
        print("fitting tfidf vectorizer...")
        vectorizer = fit_tfidf(train_texts, config)
        save_vectorizer(vectorizer, VECTORIZER_PATH)

    X_train = transform(vectorizer, train_texts)
    X_val = transform(vectorizer, val_texts)
    X_test = transform(vectorizer, test_texts)

    return X_train, X_val, X_test, vectorizer

def train_and_log(
    model_class,
    model_name: str,
    save_path: str,
    s3_prefix: str,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    X_test,
    y_test: np.ndarray,
    label_names: list[str],
    config: dict,
) -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_id = get_or_create_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=model_name,
    ) as run:
        log_config(config)

        print(f"  training {model_name}...")
        model_config = config["model"]
        model = model_class(
            C=get_config_value(model_config, "C"),
            max_iter=get_config_value(model_config, "max_iter"),
            random_state=get_config_value(model_config, "random_state"),
        )
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_metrics = compute_classification_metrics(y_val, val_preds)
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        log_metrics(val_metrics)

        test_preds = model.predict(X_test)
        test_metrics = compute_classification_metrics(y_test, test_preds)
        test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
        log_metrics(test_metrics)

        latency = compute_latency(model.predict, X_test[:100])
        log_metrics(latency)

        report = get_classification_report(y_test, test_preds, label_names)
        report_path = f"artifacts/{model_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        log_confusion_matrix(
            y_test,
            test_preds,
            label_names,
            save_path=f"artifacts/{model_name}_confusion_matrix.png",
        )

        model.save(save_path)
        mlflow.log_artifact(save_path)

        upload_artifact(save_path, f"{s3_prefix}/{save_path.split('/')[-1]}")
        upload_artifact(VECTORIZER_PATH, f"{s3_prefix}/tfidf.pkl")

        all_metrics = {**val_metrics, **test_metrics, **latency}
        print(f"  val accuracy : {val_metrics['val_accuracy']}")
        print(f"  test accuracy : {test_metrics['test_accuracy']}")
        print(f"  test macro_f1 : {test_metrics['test_macro_f1']}")
        print(f"  latency p50 : {latency['latency_p50_ms']}ms")
        print(f"  run id : {run.info.run_id}")

        return all_metrics

def main():
    config = load_config("classical")

    processed, label_map = load_or_download_data(config)

    X_train, X_val, X_test, vectorizer = get_features(processed, config)

    y_train = processed["train"]["label_id"].values
    y_val = processed["validation"]["label_id"].values
    y_test = processed["test"]["label_id"].values

    id_to_label = {v: k for k, v in label_map.items()}
    label_names = [id_to_label[i] for i in range(len(label_map))]

    print("\ntraining logistic regression...")
    logreg_metrics = train_and_log(
        model_class=LogisticRegressionModel,
        model_name="logistic-regression",
        save_path=LOGREG_PATH,
        s3_prefix=config["s3"]["prefix"],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        config=config,
    )

    config["model"]["type"] = "svm"
    print("\ntraining svm...")
    svm_metrics = train_and_log(
        model_class=SVMModel,
        model_name="svm",
        save_path=SVM_PATH,
        s3_prefix=config["s3"]["prefix"],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        config=config,
    )

    print("\nmodel comparison:")
    print(f"{'model':<25} {'val_acc':<12} {'test_acc':<12} {'macro_f1':<12} {'p50_ms'}")
    print("-" * 70)
    print(
        f"{'logistic-regression':<25} "
        f"{logreg_metrics['val_accuracy']:<12} "
        f"{logreg_metrics['test_accuracy']:<12} "
        f"{logreg_metrics['test_macro_f1']:<12} "
        f"{logreg_metrics['latency_p50_ms']}ms"
    )
    print(
        f"{'svm':<25} "
        f"{svm_metrics['val_accuracy']:<12} "
        f"{svm_metrics['test_accuracy']:<12} "
        f"{svm_metrics['test_macro_f1']:<12} "
        f"{svm_metrics['latency_p50_ms']}ms"
    )


if __name__ == "__main__":
    main()
