import json
import re
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_label_map(train_df: pd.DataFrame, label_col: str = "label") -> dict[str, int]:
    labels = sorted(train_df[label_col].unique().tolist())
    return {label: idx for idx, label in enumerate(labels)}


def encode_labels(
    df: pd.DataFrame,
    label_map: dict[str, int],
    label_col: str = "label",
) -> pd.DataFrame:
    df = df.copy()
    df["label_id"] = df[label_col].map(label_map)
    return df


def save_label_map(label_map: dict[str, int], save_path: str = "artifacts/label_map.json") -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2)


def load_label_map(load_path: str = "artifacts/label_map.json") -> dict[str, int]:
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_id_to_label(label_map: dict[str, int]) -> dict[int, str]:
    return {idx: label for label, idx in label_map.items()}


def preprocess(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    label_map = build_label_map(splits["train"])

    processed = {}
    for split_name, df in splits.items():
        df = df.copy()
        df["text"] = df["text"].apply(clean_text)
        df = encode_labels(df, label_map)
        processed[split_name] = df

    save_label_map(label_map)

    return processed, label_map
