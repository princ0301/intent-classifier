import os

from src.storage.s3 import download_artifact

os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/vectorizers", exist_ok=True)

download_artifact("classical/logreg.pkl", "artifacts/models/logreg.pkl")
download_artifact("classical/svm.pkl", "artifacts/models/svm.pkl")
download_artifact("classical/tfidf.pkl", "artifacts/vectorizers/tfidf.pkl")

print("artifacts downloaded")