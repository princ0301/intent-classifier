"""
Serve/inference for the fine-tuned DistilBERT intent classifier on Modal.

Loads the model saved by modal_train.py (artifacts/models/distilbert on the
shared Volume) and exposes:
  - predict.remote(texts)   -> callable from any Python client
  - a FastAPI web endpoint  -> POST /predict {"texts": [...]}

Usage:
    modal run modal_inference.py --texts "cancel my order" "what is my balance"
    modal deploy modal_inference.py   # to get a persistent HTTPS endpoint
"""

import modal

app = modal.App("intent-classifier-inference")

volume = modal.Volume.from_name("intent-classifier-data", create_if_missing=False)
VOLUME_PATH = "/vol"
MODEL_DIR = f"{VOLUME_PATH}/artifacts/models/distilbert"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "transformers==4.44.2",
    "numpy==1.26.4",
    "fastapi==0.115.0",
)


@app.cls(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    scaledown_window=300,
)
class IntentClassifier:

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()

        self.id_to_label = {v: k for k, v in self.model.config.label2id.items()}

    @modal.method()
    def predict(self, texts: list[str], max_length: int = 128) -> list[dict]:
        import torch

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1)

        top_probs, top_ids = probs.max(dim=1)

        results = []
        for text, label_id, prob in zip(texts, top_ids.tolist(), top_probs.tolist()):
            results.append({
                "text": text,
                "intent": self.id_to_label[label_id],
                "confidence": round(prob, 4),
            })
        return results

    @modal.fastapi_endpoint(method="POST")
    def predict_endpoint(self, payload: dict):
        texts = payload.get("texts", [])
        if not texts:
            return {"error": "payload must include a non-empty 'texts' list"}
        return {"predictions": self.predict.local(texts)}


@app.local_entrypoint()
def main(texts: list[str] = None):
    if not texts:
        texts = [
            "i want to cancel my subscription",
            "what's the weather like tomorrow",
            "can you transfer money to my savings account",
        ]
    classifier = IntentClassifier()
    results = classifier.predict.remote(texts)
    for r in results:
        print(f"{r['text']!r:60s} -> {r['intent']:30s} ({r['confidence']:.2%})")