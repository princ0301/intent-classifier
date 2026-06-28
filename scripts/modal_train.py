"""
Fine-tune DistilBERT intent classifier on Modal.

Mounts the local repo into the container, runs train_transformer.main()
on a remote GPU, and persists data/ and artifacts/ to a Modal Volume so
downloaded data and trained models survive across runs.

Setup (one-time):
    modal secret create mlflow-secret MLFLOW_TRACKING_URI=<your-uri>
    modal secret create aws-secret AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret>

Usage:
    modal run modal_train.py
"""

import modal

app = modal.App("intent-classifier-train")

volume = modal.Volume.from_name("intent-classifier-data", create_if_missing=True)
VOLUME_PATH = "/vol"

GPU = "A10G"
TIMEOUT_SECONDS = 60 * 60 * 2

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "datasets==2.21.0",
        "scikit-learn==1.5.1",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "mlflow==2.16.0",
        "boto3==1.35.0",
        "pyyaml==6.0.2",
        "matplotlib==3.9.0",
    )
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=["data", "artifacts", ".git", "__pycache__", "*.pyc"],
    )
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("mlflow-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
)
def train():
    import os
    import sys

    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")

    vol_data_dir = os.path.join(VOLUME_PATH, "data", "raw")
    vol_artifacts_dir = os.path.join(VOLUME_PATH, "artifacts")
    os.makedirs(vol_data_dir, exist_ok=True)
    os.makedirs(vol_artifacts_dir, exist_ok=True)

    os.makedirs("data", exist_ok=True)
    os.symlink(vol_data_dir, "data/raw")
    os.symlink(vol_artifacts_dir, "artifacts")

    from train_transformer import main as run_training

    run_training()

    volume.commit()


@app.local_entrypoint()
def main():
    train.remote()