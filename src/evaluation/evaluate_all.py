import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from src.data.loader import load_splits
from src.data.preprocessor import preprocess
from src.features.tfidf import load_vectorizer, transform
from src.models.classical import LogisticRegressionModel, SVMModel
from src.models.neural import IntentDatasetNN, LSTMModel, TextCNN, Vocabulary
from src.models.transformer import IntentDatasetHF, TransformerModel
from src.utils.config import load_config
from src.utils.mlflow_utils import get_or_create_experiment
from src.utils.settings import settings

VECTORIZER_PATH = "artifacts/vectorizers/tfidf.pkl"
VOCAB_PATH = "artifacts/models/vocab.pkl"
LOGREG_PATH = "artifacts/models/logreg.pkl"
SVM_PATH = "artifacts/models/svm.pkl"
TEXTCNN_PATH = "artifacts/models/textcnn.pt"
RNN_PATH = "artifacts/models/rnn.pt"
LSTM_PATH = "artifacts/models/lstm.pt"
DISTILBERT_DIR = "artifacts/models/distilbert"
MAX_LENGTH_NN = 32
MAX_LENGTH_HF = 128
REPORT_DIR = Path("artifacts/evaluation")


def load_data():
    splits = load_splits("data/raw")
    processed, label_map = preprocess(splits)
    id_to_label = {v: k for k, v in label_map.items()}
    label_names = [id_to_label[i] for i in range(len(label_map))]
    return processed, label_map, label_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def measure_latency(predict_fn, input_data, n_runs: int = 50) -> dict:
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(input_data)
        latencies.append((time.perf_counter() - start) * 1000)
    latencies = np.array(latencies)
    return {
        "latency_mean_ms": round(float(np.mean(latencies)), 3),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 3),
    }


def eval_logreg(processed: dict, label_names: list[str]) -> tuple[dict, np.ndarray]:
    print("  evaluating logistic regression...")
    vectorizer = load_vectorizer(VECTORIZER_PATH)
    X_test = transform(vectorizer, processed["test"]["text"].tolist())
    y_test = processed["test"]["label_id"].values

    model = LogisticRegressionModel()
    model.load(LOGREG_PATH)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    latency = measure_latency(model.predict, X_test[:100])
    return {**metrics, **latency}, y_pred


def eval_svm(processed: dict, label_names: list[str]) -> tuple[dict, np.ndarray]:
    print("  evaluating svm...")
    vectorizer = load_vectorizer(VECTORIZER_PATH)
    X_test = transform(vectorizer, processed["test"]["text"].tolist())
    y_test = processed["test"]["label_id"].values

    model = SVMModel()
    model.load(SVM_PATH)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    latency = measure_latency(model.predict, X_test[:100])
    return {**metrics, **latency}, y_pred


def eval_neural(
    model_type: str,
    model_path: str,
    processed: dict,
    label_map: dict,
) -> tuple[dict, np.ndarray]:
    print(f"  evaluating {model_type}...")
    vocab = Vocabulary.load(VOCAB_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_map)

    neural_config = load_config("neural")
    model_cfg = neural_config["model"][model_type]

    if model_type == "textcnn":
        model = TextCNN(
            vocab_size=len(vocab),
            embedding_dim=model_cfg["embedding_dim"],
            num_filters=model_cfg["num_filters"],
            kernel_sizes=model_cfg["kernel_sizes"],
            num_classes=num_classes,
            dropout=model_cfg["dropout"],
        )
    elif model_type == "rnn":
        from src.models.neural import RNNModel

        model = RNNModel(
            vocab_size=len(vocab),
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_classes=num_classes,
            dropout=model_cfg["dropout"],
        )
    elif model_type == "lstm":
        model = LSTMModel(
            vocab_size=len(vocab),
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_classes=num_classes,
            dropout=model_cfg["dropout"],
        )

    model.load(model_path)
    model.to(device)
    model.eval()

    dataset = IntentDatasetNN(
        processed["test"]["text"].tolist(),
        processed["test"]["label_id"].tolist(),
        vocab,
        MAX_LENGTH_NN,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    y_test = processed["test"]["label_id"].values

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    y_pred = np.array(all_preds)
    metrics = compute_metrics(y_test, y_pred)

    def predict_fn(loader):
        with torch.no_grad():
            for inputs, _ in loader:
                model(inputs.to(device))

    latency = measure_latency(predict_fn, loader, n_runs=20)
    return {**metrics, **latency}, y_pred


def eval_distilbert(processed: dict, label_map: dict) -> tuple[dict, np.ndarray]:
    print("  evaluating distilbert...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = TransformerModel(
        model_name=DISTILBERT_DIR,
        num_labels=len(label_map),
    )
    transformer.model.to(device)
    transformer.model.eval()

    dataset = IntentDatasetHF(
        processed["test"]["text"].tolist(),
        processed["test"]["label_id"].tolist(),
        transformer.tokenizer,
        MAX_LENGTH_HF,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    y_test = processed["test"]["label_id"].values

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = transformer.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    y_pred = np.array(all_preds)
    metrics = compute_metrics(y_test, y_pred)

    single_text = processed["test"]["text"].iloc[:1].tolist()
    single_dataset = IntentDatasetHF(
        single_text,
        [0],
        transformer.tokenizer,
        MAX_LENGTH_HF,
    )
    single_loader = DataLoader(single_dataset, batch_size=1)

    def predict_fn(loader):
        with torch.no_grad():
            for batch in loader:
                transformer.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )

    latency = measure_latency(predict_fn, single_loader, n_runs=50)
    return {**metrics, **latency}, y_pred


def plot_comparison(results: dict, save_path: str) -> None:
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["model"] + list(df.columns[1:])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    titles = ["Accuracy", "Macro F1", "Weighted F1"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(df["model"], df[metric].astype(float), color=colors)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(len(df["model"])))
        ax.set_xticklabels(df["model"], rotation=30, ha="right")
        for bar, val in zip(bars, df[metric].astype(float)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle("Model Comparison — CLINC150", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def plot_latency(results: dict, save_path: str) -> None:
    models = list(results.keys())
    p50 = [results[m]["latency_p50_ms"] for m in models]
    p95 = [results[m]["latency_p95_ms"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, p50, width, label="P50", color="#4C72B0")
    ax.bar(x + width / 2, p95, width, label="P95", color="#DD8452")
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms) — log scale")
    ax.set_title("Inference Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def get_top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                pairs.append(
                    {
                        "true_label": label_names[i],
                        "predicted_label": label_names[j],
                        "count": int(cm[i, j]),
                    }
                )

    df = pd.DataFrame(pairs).sort_values("count", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def plot_top_confused_pairs(df: pd.DataFrame, model_name: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
    labels = [f"{row.true_label} -> {row.predicted_label}" for row in df.itertuples()]
    ax.barh(labels, df["count"], color="#C44E52")
    ax.invert_yaxis()
    ax.set_xlabel("Misclassification Count")
    ax.set_title(f"Top Confused Pairs — {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def plot_oos_binary_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict,
    model_name: str,
    save_path: str,
) -> None:
    oos_id = label_map.get("oos")
    y_true_binary = (y_true == oos_id).astype(int)
    y_pred_binary = (y_pred == oos_id).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["in-scope", "oos"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"OOS Detection — {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading data...")
    processed, label_map, label_names = load_data()
    y_test = processed["test"]["label_id"].values

    print("\nevaluating all models...")
    results = {}
    predictions = {}

    results["logreg"], predictions["logreg"] = eval_logreg(processed, label_names)
    results["svm"], predictions["svm"] = eval_svm(processed, label_names)
    results["textcnn"], predictions["textcnn"] = eval_neural("textcnn", TEXTCNN_PATH, processed, label_map)
    results["rnn"], predictions["rnn"] = eval_neural("rnn", RNN_PATH, processed, label_map)
    results["lstm"], predictions["lstm"] = eval_neural("lstm", LSTM_PATH, processed, label_map)
    results["distilbert"], predictions["distilbert"] = eval_distilbert(processed, label_map)

    print("\ngenerating plots...")
    plot_comparison(results, str(REPORT_DIR / "model_comparison.png"))
    plot_latency(results, str(REPORT_DIR / "latency_comparison.png"))

    for model_name, y_pred in predictions.items():
        confused_df = get_top_confused_pairs(y_test, y_pred, label_names, top_n=20)
        confused_csv_path = REPORT_DIR / f"{model_name}_top_confused_pairs.csv"
        confused_df.to_csv(confused_csv_path, index=False)
        print(f"  saved: {confused_csv_path}")

        plot_top_confused_pairs(
            confused_df,
            model_name,
            str(REPORT_DIR / f"{model_name}_top_confused_pairs.png"),
        )

        plot_oos_binary_confusion(
            y_test,
            y_pred,
            label_map,
            model_name,
            str(REPORT_DIR / f"{model_name}_oos_binary.png"),
        )

    print("\nlogging to MLflow...")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_id = get_or_create_experiment("intent-classifier")

    with mlflow.start_run(experiment_id=experiment_id, run_name="unified-evaluation"):
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{model_name}.{metric_name}", value)

        mlflow.log_artifact(str(REPORT_DIR / "model_comparison.png"))
        mlflow.log_artifact(str(REPORT_DIR / "latency_comparison.png"))

    print("\nfinal results:")
    print(f"{'model':<14} {'accuracy':<12} {'macro_f1':<12} {'weighted_f1':<14} {'p50_ms':<12} {'p95_ms'}")
    print("-" * 78)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<14} "
            f"{metrics['accuracy']:<12} "
            f"{metrics['macro_f1']:<12} "
            f"{metrics['weighted_f1']:<14} "
            f"{metrics['latency_p50_ms']:<12} "
            f"{metrics['latency_p95_ms']}"
        )


if __name__ == "__main__":
    main()
