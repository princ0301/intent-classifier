from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_clinc150(subset: str = "plus") -> dict[str, pd.DataFrame]:
    dataset = load_dataset("clinc/clinc_oos", subset)

    intent_feature = dataset["train"].features["intent"]

    splits = {}
    for split_name in ["train", "validation", "test"]:
        records = [
            {
                "text": item["text"],
                "label": intent_feature.int2str(item["intent"]),
            }
            for item in dataset[split_name]
        ]
        splits[split_name] = pd.DataFrame(records)

    return splits


def save_splits(splits: dict[str, pd.DataFrame], save_dir: str = "data/raw") -> None:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    for split_name, df in splits.items():
        df.to_csv(path / f"{split_name}.csv", index=False)


def load_splits(load_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    path = Path(load_dir)
    split_names = ["train", "validation", "test"]

    splits = {}
    for split_name in split_names:
        file_path = path / f"{split_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Split not found: {file_path}")
        splits[split_name] = pd.read_csv(file_path)

    return splits
