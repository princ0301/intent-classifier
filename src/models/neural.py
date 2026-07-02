import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2token = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts: list[str]) -> None:
        freq = {}
        for text in texts:
            for token in text.split():
                freq[token] = freq.get(token, 0) + 1

        for token, count in freq.items():
            if count >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def encode(self, text: str, max_length: int) -> list[int]:
        tokens = text.split()[:max_length]
        ids = [self.token2idx.get(t, 1) for t in tokens]
        ids += [0] * (max_length - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2idx)

    def save(self, save_path: str) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(load_path: str) -> "Vocabulary":
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)


class IntentDatasetNN(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        max_length: int = 32,
    ):
        self.labels = labels
        self.encodings = [vocab.encode(text, max_length) for text in texts]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: list[int],
        num_classes: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)

        pooled = []
        for conv in self.convs:
            activated = torch.relu(conv(embedded))
            pool = torch.max(activated, dim=2).values
            pooled.append(pool)

        concatenated = torch.cat(pooled, dim=1)
        dropped = self.dropout(concatenated)
        return self.fc(dropped)

    def save(self, save_path: str) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, load_path: str) -> None:
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.rnn(embedded)
        out = self.dropout(hidden[-1])
        return self.fc(out)

    def save(self, save_path: str) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, load_path: str) -> None:
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        return self.fc(out)

    def save(self, save_path: str) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, load_path: str) -> None:
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
