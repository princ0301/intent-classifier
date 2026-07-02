from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class IntentDatasetHF(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TransformerModel:
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            seq_classif_dropout=dropout,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def load(self, load_dir: str) -> None:
        path = Path(load_dir)
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)

    def predict_proba(self, texts: list[str], max_length: int = 128) -> np.ndarray:
        self.model.eval()
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)
