from src.data.loader import load_clinc150, save_splits
from src.data.preprocessor import preprocess
from src.features.tfidf import fit_tfidf, transform, save_vectorizer, load_vectorizer, get_feature_names
from src.utils.config import load_config


def main():
    print("loading data...")
    splits = load_clinc150(subset="plus")
    save_splits(splits)

    print("preprocessing...")
    processed, label_map = preprocess(splits)

    train_texts = processed["train"]["text"].tolist()
    val_texts = processed["validation"]["text"].tolist()
    test_texts = processed["test"]["text"].tolist()

    print("fitting tfidf on train only...")
    config = load_config("classical")
    vectorizer = fit_tfidf(train_texts, config)

    print("transforming splits...")
    X_train = transform(vectorizer, train_texts)
    X_val = transform(vectorizer, val_texts)
    X_test = transform(vectorizer, test_texts)

    print(f"\nfeature matrix shapes:")
    print(f"  train : {X_train.shape}")
    print(f"  val   : {X_val.shape}")
    print(f"  test  : {X_test.shape}")

    vocab_size = len(get_feature_names(vectorizer))
    print(f"\nvocabulary size : {vocab_size}")

    print("\nsaving vectorizer...")
    save_vectorizer(vectorizer)

    print("loading vectorizer from disk...")
    loaded = load_vectorizer()
    X_check = transform(loaded, train_texts[:5])
    assert X_check.shape[1] == X_train.shape[1]
    print("vectorizer round-trip check passed")

    print("\ntop 20 features by idf score (most unique/informative):")
    import numpy as np
    feature_names = get_feature_names(vectorizer)
    idf_scores = vectorizer.idf_
    top_indices = np.argsort(idf_scores)[-20:][::-1]
    for idx in top_indices:
        print(f"  {feature_names[idx]:<30} idf={idf_scores[idx]:.4f}")

    print("\nphase 3 complete")


if __name__ == "__main__":
    main()