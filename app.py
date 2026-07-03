import os
import subprocess
import threading
import time


def start_api():
    subprocess.run(
        [
            "uvicorn",
            "api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
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

    download_artifact("classical/logreg.pkl", "artifacts/models/logreg.pkl")
    download_artifact("classical/svm.pkl", "artifacts/models/svm.pkl")
    download_artifact("classical/tfidf.pkl", "artifacts/vectorizers/tfidf.pkl")
    download_artifact("neural/vocab.pkl", "artifacts/models/vocab.pkl")
    download_artifact("transformer/distilbert/config.json", "artifacts/models/distilbert/config.json")
    download_artifact("transformer/distilbert/model.safetensors", "artifacts/models/distilbert/model.safetensors")
    download_artifact("transformer/distilbert/tokenizer.json", "artifacts/models/distilbert/tokenizer.json")
    download_artifact("transformer/distilbert/tokenizer_config.json", "artifacts/models/distilbert/tokenizer_config.json")
    download_artifact("transformer/distilbert/special_tokens_map.json", "artifacts/models/distilbert/special_tokens_map.json")
    download_artifact("transformer/distilbert/vocab.txt", "artifacts/models/distilbert/vocab.txt")

    print("models downloaded")


def generate_label_map():
    from src.data.loader import load_clinc150, save_splits
    from src.data.preprocessor import preprocess

    label_map_path = "artifacts/label_map.json"
    if os.path.exists(label_map_path):
        print("label map already exists")
        return

    print("generating label map...")
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

    os.execv(
        "/usr/bin/streamlit",
        [
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.port=7860",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ],
    )