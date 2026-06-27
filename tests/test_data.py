import pytest
import pandas as pd
from src.data.preprocessor import (
    clean_text,
    build_label_map,
    encode_labels,
    save_label_map,
    load_label_map,
    get_id_to_label,
    preprocess,
)


@pytest.fixture
def sample_splits():
    train = pd.DataFrame({
        "text": ["what is my balance", "transfer money please", "book a flight"],
        "label": ["balance", "transfer", "flight"],
    })
    val = pd.DataFrame({
        "text": ["show my balance"],
        "label": ["balance"],
    })
    test = pd.DataFrame({
        "text": ["send money", "unknown request"],
        "label": ["transfer", "oos"],
    })
    return {"train": train, "validation": val, "test": test}


def test_clean_text_lowercase():
    assert clean_text("Hello World") == "hello world"


def test_clean_text_strips_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_collapses_spaces():
    assert clean_text("hello    world") == "hello world"


def test_build_label_map_sorted(sample_splits):
    label_map = build_label_map(sample_splits["train"])
    assert list(label_map.keys()) == sorted(label_map.keys())


def test_build_label_map_unique_ids(sample_splits):
    label_map = build_label_map(sample_splits["train"])
    ids = list(label_map.values())
    assert len(ids) == len(set(ids))


def test_encode_labels_no_nulls(sample_splits):
    label_map = build_label_map(sample_splits["train"])
    encoded = encode_labels(sample_splits["train"], label_map)
    assert encoded["label_id"].isna().sum() == 0


def test_no_label_leakage(sample_splits):
    label_map = build_label_map(sample_splits["train"])
    train_labels = set(label_map.keys())
    val_labels = set(sample_splits["validation"]["label"].unique())
    assert val_labels.issubset(train_labels)


def test_get_id_to_label_roundtrip(sample_splits):
    label_map = build_label_map(sample_splits["train"])
    id_to_label = get_id_to_label(label_map)
    for label, idx in label_map.items():
        assert id_to_label[idx] == label


def test_save_and_load_label_map(tmp_path, sample_splits):
    label_map = build_label_map(sample_splits["train"])
    save_path = str(tmp_path / "label_map.json")
    save_label_map(label_map, save_path)
    loaded = load_label_map(save_path)
    assert loaded == label_map


def test_preprocess_returns_all_splits(sample_splits, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processed, label_map = preprocess(sample_splits)
    assert set(processed.keys()) == {"train", "validation", "test"}


def test_preprocess_adds_label_id_column(sample_splits, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processed, _ = preprocess(sample_splits)
    for df in processed.values():
        assert "label_id" in df.columns


def test_preprocess_text_is_cleaned(sample_splits, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processed, _ = preprocess(sample_splits)
    for df in processed.values():
        for text in df["text"]:
            assert text == text.lower()
            assert text == text.strip()