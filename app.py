import os
import subprocess
import sys
import threading
import time


def start_api():
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        check=False,
    )


def wait_for_api(timeout: int = 120) -> bool:
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get("http://localhost:8000/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def download_models():
    from src.storage.s3 import download_artifact

    print("downloading models from S3...")

    os.makedirs("artifacts/models/distilbert", exist_ok=True)
    os.makedirs("artifacts/vectorizers", exist_ok=True)

    files = [
        ("classical/logreg.pkl", "artifacts/models/logreg.pkl"),
        ("classical/svm.pkl", "artifacts/models/svm.pkl"),
        ("classical/tfidf.pkl", "artifacts/vectorizers/tfidf.pkl"),
        ("neural/vocab.pkl", "artifacts/models/vocab.pkl"),
        ("transformer/distilbert/config.json", "artifacts/models/distilbert/config.json"),
        ("transformer/distilbert/model.safetensors", "artifacts/models/distilbert/model.safetensors"),
        ("transformer/distilbert/tokenizer.json", "artifacts/models/distilbert/tokenizer.json"),
        ("transformer/distilbert/tokenizer_config.json", "artifacts/models/distilbert/tokenizer_config.json"),
        ("transformer/distilbert/special_tokens_map.json", "artifacts/models/distilbert/special_tokens_map.json"),
        ("transformer/distilbert/vocab.txt", "artifacts/models/distilbert/vocab.txt"),
    ]

    for s3_key, local_path in files:
        if not os.path.exists(local_path):
            download_artifact(s3_key, local_path)

    print("models downloaded")


def generate_label_map():
    if os.path.exists("artifacts/label_map.json"):
        print("label map already exists")
        return

    print("generating label map...")
    from src.data.loader import load_clinc150, save_splits
    from src.data.preprocessor import preprocess

    splits = load_clinc150("plus")
    save_splits(splits)
    preprocess(splits)
    print("label map generated")


if __name__ == "__main__":
    download_models()
    generate_label_map()

    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()

    print("waiting for API to start...")
    if wait_for_api():
        print("API ready")
    else:
        print("API startup timed out — continuing anyway")

    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            "app/streamlit_app.py",
            "--server.port=7860",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ],
        check=True,
    )