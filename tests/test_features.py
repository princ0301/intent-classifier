import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.tfidf import (
    fit_tfidf,
    transform,
    save_vectorizer,
    load_vectorizer,
    get_feature_names,
)


@pytest.fixture
def sample_texts():
    return [
        "what is my account balance",
        "transfer money to savings",
        "book a flight to new york",
        "what is the weather today",
        "set an alarm for tomorrow morning",
    ]


@pytest.fixture
def sample_config():
    return {
        "tfidf": {
            "max_features": 100,
            "ngram_range": [1, 2],
            "min_df": 1,
        }
    }


@pytest.fixture
def fitted_vectorizer(sample_texts, sample_config):
    return fit_tfidf(sample_texts, sample_config)


def test_fit_returns_vectorizer(fitted_vectorizer):
    assert isinstance(fitted_vectorizer, TfidfVectorizer)


def test_vocabulary_size_respects_max_features(sample_texts, sample_config):
    vectorizer = fit_tfidf(sample_texts, sample_config)
    assert len(get_feature_names(vectorizer)) <= sample_config["tfidf"]["max_features"]


def test_transform_returns_correct_shape(fitted_vectorizer, sample_texts):
    X = transform(fitted_vectorizer, sample_texts)
    assert X.shape[0] == len(sample_texts)


def test_transform_columns_match_vocabulary(fitted_vectorizer, sample_texts):
    X = transform(fitted_vectorizer, sample_texts)
    assert X.shape[1] == len(get_feature_names(fitted_vectorizer))


def test_no_fit_on_val_or_test(sample_texts, sample_config):
    train = sample_texts[:3]
    val = sample_texts[3:]
    vectorizer = fit_tfidf(train, sample_config)
    X_train = transform(vectorizer, train)
    X_val = transform(vectorizer, val)
    assert X_train.shape[1] == X_val.shape[1]


def test_save_and_load_vectorizer(fitted_vectorizer, sample_texts, tmp_path):
    save_path = str(tmp_path / "tfidf.pkl")
    save_vectorizer(fitted_vectorizer, save_path)
    loaded = load_vectorizer(save_path)
    X_original = transform(fitted_vectorizer, sample_texts)
    X_loaded = transform(loaded, sample_texts)
    assert np.allclose(X_original.toarray(), X_loaded.toarray())


def test_load_vectorizer_missing_file():
    with pytest.raises(FileNotFoundError):
        load_vectorizer("nonexistent/path/tfidf.pkl")


def test_feature_names_returns_list(fitted_vectorizer):
    names = get_feature_names(fitted_vectorizer)
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


def test_tfidf_values_are_nonnegative(fitted_vectorizer, sample_texts):
    X = transform(fitted_vectorizer, sample_texts)
    assert X.min() >= 0