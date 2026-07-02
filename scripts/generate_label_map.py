from src.data.loader import load_clinc150, save_splits
from src.data.preprocessor import preprocess

splits = load_clinc150("plus")
save_splits(splits)
preprocess(splits)
print("label map generated")