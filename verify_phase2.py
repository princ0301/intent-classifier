from src.data.loader import load_clinc150, save_splits
from src.data.preprocessor import preprocess, load_label_map, get_id_to_label


def main():
    print("loading CLINC150...")
    splits = load_clinc150(subset="plus")

    print("saving raw splits...")
    save_splits(splits)

    print("split sizes:")
    for name, df in splits.items():
        print(f"  {name}: {len(df)} rows, {df['label'].nunique()} unique intents")

    print("\npreprocessing...")
    processed, label_map = preprocess(splits)

    print(f"label map size: {len(label_map)} intents")
    print(f"label map saved to artifacts/label_map.json")

    id_to_label = get_id_to_label(label_map)

    print("\nsample rows from train:")
    for _, row in processed["train"].head(3).iterrows():
        print(f"  text     : {row['text']}")
        print(f"  label    : {row['label']}")
        print(f"  label_id : {row['label_id']} -> {id_to_label[row['label_id']]}")
        print()

    print("loading label map from disk...")
    loaded_map = load_label_map()
    assert len(loaded_map) == len(label_map)
    print("label map round-trip check passed")

    oos_count = (processed["test"]["label"] == "oos").sum()
    print(f"\nout-of-scope samples in test set: {oos_count}")

    print("\nphase 2 complete")


if __name__ == "__main__":
    main()