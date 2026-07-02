import pickle
from pathlib import Path

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def _get_tfidf_value(tfidf_config: dict, key: str):
    kebab_key = key.replace("_", "-")
    if key in tfidf_config:
        return tfidf_config[key]
    if kebab_key in tfidf_config:
        return tfidf_config[kebab_key]
    raise KeyError(key)


def fit_tfidf(train_texts: list[str], config: dict) -> TfidfVectorizer:
    tfidf_config = config["tfidf"]
    vectorizer = TfidfVectorizer(
        max_features=_get_tfidf_value(tfidf_config, "max_features"),
        ngram_range=tuple(_get_tfidf_value(tfidf_config, "ngram_range")),
        min_df=_get_tfidf_value(tfidf_config, "min_df"),
    )
    vectorizer.fit(train_texts)
    return vectorizer


def transform(vectorizer: TfidfVectorizer, texts: list[str]) -> sp.csr_matrix:
    return vectorizer.transform(texts)


def save_vectorizer(
    vectorizer: TfidfVectorizer, save_path: str = "artifacts/vectorizers/tfidf.pkl"
) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(
    load_path: str = "artifacts/vectorizers/tfidf.pkl",
) -> TfidfVectorizer:
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_names(vectorizer: TfidfVectorizer) -> list[str]:
    return vectorizer.get_feature_names_out().tolist()
