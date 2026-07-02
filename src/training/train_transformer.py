import os

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from src.data.loader import load_clinc150, load_splits, save_splits
from src.data.preprocessor import preprocess
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_latency,
    get_classification_report,
)
from src.models.transformer import IntentDatasetHF, TransformerModel
from src.storage.s3 import upload_artifact
from src.utils.config import load_config
from src.utils.mlflow_utils import (
    get_or_create_experiment,
    log_confusion_matrix,
    log_metrics,
)
from src.utils.settings import settings

DATA_DIR = "data/raw"
MODEL_DIR = "artifacts/models/distilbert"


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


def compute_metrics_hf(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    metrics = compute_classification_metrics(labels, preds)
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def predict_all(
    transformer: TransformerModel,
    dataset: IntentDatasetHF,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    transformer.model.eval()
    transformer.model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = transformer.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    config = load_config("transformer")
    torch.manual_seed(config["training"]["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    processed, label_map = load_or_download_data(config)

    model_cfg = config["model"]
    training_cfg = config["training"]
    max_length = config["data"]["max_length"]

    print(f"loading {model_cfg['name']}...")
    transformer = TransformerModel(
        model_name=model_cfg["name"],
        num_labels=model_cfg["num_labels"],
        dropout=model_cfg["dropout"],
    )

    print("tokenizing datasets...")
    train_dataset = IntentDatasetHF(
        processed["train"]["text"].tolist(),
        processed["train"]["label_id"].tolist(),
        transformer.tokenizer,
        max_length,
    )
    val_dataset = IntentDatasetHF(
        processed["validation"]["text"].tolist(),
        processed["validation"]["label_id"].tolist(),
        transformer.tokenizer,
        max_length,
    )
    test_dataset = IntentDatasetHF(
        processed["test"]["text"].tolist(),
        processed["test"]["label_id"].tolist(),
        transformer.tokenizer,
        max_length,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_id = get_or_create_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=config["mlflow"]["run_name"],
    ) as run:
        mlflow.log_params(
            {
                "model.name": model_cfg["name"],
                "model.num_labels": model_cfg["num_labels"],
                "model.dropout": model_cfg["dropout"],
                "training.epochs": training_cfg["epochs"],
                "training.batch_size": training_cfg["batch_size"],
                "training.learning_rate": training_cfg["learning_rate"],
                "training.warmup_steps": training_cfg["warmup_steps"],
                "training.weight_decay": training_cfg["weight_decay"],
                "data.max_length": max_length,
            }
        )

        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            num_train_epochs=training_cfg["epochs"],
            per_device_train_batch_size=training_cfg["batch_size"],
            per_device_eval_batch_size=training_cfg["batch_size"],
            learning_rate=training_cfg["learning_rate"],
            warmup_steps=training_cfg["warmup_steps"],
            weight_decay=training_cfg["weight_decay"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=50,
            report_to="none",
            seed=training_cfg["random_state"],
        )

        trainer = Trainer(
            model=transformer.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_hf,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("\nfine-tuning distilbert...")
        trainer.train()

        print("\nloading best checkpoint...")
        transformer.save(MODEL_DIR)

        id_to_label = {v: k for k, v in label_map.items()}
        label_names = [id_to_label[i] for i in range(len(label_map))]

        print("evaluating on test set...")
        test_preds, test_labels = predict_all(
            transformer, test_dataset, training_cfg["batch_size"], device
        )

        test_metrics = compute_classification_metrics(test_labels, test_preds)
        test_metrics_logged = {f"test_{k}": v for k, v in test_metrics.items()}
        log_metrics(test_metrics_logged)

        def predict_fn(dataset):
            predict_all(transformer, dataset, training_cfg["batch_size"], device)

        latency = compute_latency(predict_fn, test_dataset, n_runs=20)
        log_metrics(latency)

        report = get_classification_report(test_labels, test_preds, label_names)
        report_path = "artifacts/distilbert_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        log_confusion_matrix(
            test_labels,
            test_preds,
            label_names,
            save_path="artifacts/distilbert_confusion_matrix.png",
        )

        mlflow.log_artifact(MODEL_DIR)

        print("uploading to S3...")
        for fname in os.listdir(MODEL_DIR):
            local_path = os.path.join(MODEL_DIR, fname)
            if os.path.isfile(local_path):
                upload_artifact(
                    local_path, f"{config['s3']['prefix']}/distilbert/{fname}"
                )

        print(f"\ntest accuracy : {test_metrics['accuracy']}")
        print(f"test macro_f1 : {test_metrics['macro_f1']}")
        print(f"latency p50 : {latency['latency_p50_ms']}ms")
        print(f"run id : {run.info.run_id}")


if __name__ == "__main__":
    main()
