import os

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.loader import load_clinc150, load_splits, save_splits
from src.data.preprocessor import preprocess
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_latency,
    get_classification_report,
)
from src.models.neural import IntentDatasetNN, LSTMModel, RNNModel, TextCNN, Vocabulary
from src.storage.s3 import upload_artifact
from src.utils.config import load_config
from src.utils.mlflow_utils import (
    get_or_create_experiment,
    log_confusion_matrix,
    log_metrics,
)
from src.utils.settings import settings

DATA_DIR = "data/raw"
VOCAB_PATH = "artifacts/models/vocab.pkl"
MAX_LENGTH = 32


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


def build_dataloaders(
    processed: dict,
    vocab: Vocabulary,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = IntentDatasetNN(
        processed["train"]["text"].tolist(),
        processed["train"]["label_id"].tolist(),
        vocab,
        MAX_LENGTH,
    )
    val_dataset = IntentDatasetNN(
        processed["validation"]["text"].tolist(),
        processed["validation"]["label_id"].tolist(),
        vocab,
        MAX_LENGTH,
    )
    test_dataset = IntentDatasetNN(
        processed["test"]["text"].tolist(),
        processed["test"]["label_id"].tolist(),
        vocab,
        MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
    grad_clip: float | None = None,
) -> tuple[float, float]:
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def build_model(model_type: str, model_cfg: dict, vocab_size: int, num_classes: int) -> nn.Module:
    if model_type == "textcnn":
        return TextCNN(
            vocab_size=vocab_size,
            embedding_dim=model_cfg["embedding_dim"],
            num_filters=model_cfg["num_filters"],
            kernel_sizes=model_cfg["kernel_sizes"],
            num_classes=num_classes,
            dropout=model_cfg["dropout"],
        )
    if model_type == "rnn":
        return RNNModel(
            vocab_size=vocab_size,
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_classes=num_classes,
            dropout=model_cfg["dropout"],
        )
    if model_type == "lstm":
        return LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_classes=num_classes,
            dropout=model_cfg["dropout"],
        )
    raise ValueError(f"unknown model type: {model_type}")


def train_model(
    model_type: str,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vocab: Vocabulary,
    label_map: dict,
    device: torch.device,
) -> dict:
    num_classes = len(label_map)
    model_cfg = config["model"][model_type]
    save_path = f"artifacts/models/{model_type}.pt"
    grad_clip = config["training"].get("grad_clip", None)

    model = build_model(model_type, model_cfg, len(vocab), num_classes).to(device)
    print(f"\n{model_type} parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_id = get_or_create_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(experiment_id=experiment_id, run_name=model_type) as run:
        flat_config = {
            "model.type": model_type,
            **{f"model.{k}": v for k, v in model_cfg.items()},
            **{f"training.{k}": v for k, v in config["training"].items()},
        }
        mlflow.log_params(flat_config)

        epochs = config["training"]["epochs"]
        patience = config["training"]["patience"]
        best_val_loss = float("inf")
        epochs_no_improve = 0

        print(f"training {model_type} for up to {epochs} epochs (patience={patience}, grad_clip={grad_clip})...")
        print(f"{'epoch':<8} {'train_loss':<14} {'train_acc':<14} {'val_loss':<14} {'val_acc'}")
        print("-" * 65)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                training=True,
                grad_clip=grad_clip,
            )
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                device,
                training=False,
            )

            log_metrics(
                {
                    "train_loss": round(train_loss, 4),
                    "train_accuracy": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 4),
                },
                step=epoch,
            )

            print(f"{epoch:<8} {train_loss:<14.4f} {train_acc:<14.4f} {val_loss:<14.4f} {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                model.save(save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"early stopping at epoch {epoch}")
                    break

        print("loading best checkpoint...")
        model.load(save_path)

        id_to_label = {v: k for k, v in label_map.items()}
        label_names = [id_to_label[i] for i in range(num_classes)]

        test_preds, test_labels = predict_all(model, test_loader, device)
        test_metrics = compute_classification_metrics(test_labels, test_preds)
        test_metrics_logged = {f"test_{k}": v for k, v in test_metrics.items()}
        log_metrics(test_metrics_logged)

        def predict_fn(loader):
            predict_all(model, loader, device)

        latency = compute_latency(predict_fn, test_loader, n_runs=50)
        log_metrics(latency)

        report = get_classification_report(test_labels, test_preds, label_names)
        report_path = f"artifacts/{model_type}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        log_confusion_matrix(
            test_labels,
            test_preds,
            label_names,
            save_path=f"artifacts/{model_type}_confusion_matrix.png",
        )

        mlflow.log_artifact(save_path)
        upload_artifact(save_path, f"{config['s3']['prefix']}/{model_type}.pt")

        print(f"  test accuracy : {test_metrics['accuracy']}")
        print(f"  test macro_f1 : {test_metrics['macro_f1']}")
        print(f"  latency p50   : {latency['latency_p50_ms']}ms")
        print(f"  run id        : {run.info.run_id}")

        return {**test_metrics_logged, **latency}


def main():
    config = load_config("neural")
    torch.manual_seed(config["training"]["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    processed, label_map = load_or_download_data(config)

    print("building vocabulary on train set only...")
    vocab = Vocabulary()
    vocab.build(processed["train"]["text"].tolist())
    vocab.save(VOCAB_PATH)
    print(f"vocabulary size: {len(vocab)}")

    batch_size = config["training"]["batch_size"]
    train_loader, val_loader, test_loader = build_dataloaders(processed, vocab, batch_size)

    all_results = {}
    for model_type in ["textcnn", "rnn", "lstm"]:
        metrics = train_model(
            model_type=model_type,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            vocab=vocab,
            label_map=label_map,
            device=device,
        )
        all_results[model_type] = metrics

    print("\n\nfinal comparison:")
    print(f"{'model':<12} {'test_acc':<12} {'macro_f1':<12} {'p50_ms'}")
    print("-" * 50)
    for model_type, metrics in all_results.items():
        print(
            f"{model_type:<12} "
            f"{metrics['test_accuracy']:<12} "
            f"{metrics['test_macro_f1']:<12} "
            f"{metrics['latency_p50_ms']}ms"
        )


if __name__ == "__main__":
    main()
