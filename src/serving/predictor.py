import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.data.preprocessor import clean_text, load_label_map
from src.features.tfidf import load_vectorizer, transform
from src.models.classical import LogisticRegressionModel, SVMModel
from src.models.neural import LSTMModel, RNNModel, TextCNN, Vocabulary
from src.models.transformer import TransformerModel
from src.utils.config import load_config
from src.utils.settings import settings

VECTORIZER_PATH = "artifacts/vectorizers/tfidf.pkl"
VOCAB_PATH = "artifacts/models/vocab.pkl"
LOGREG_PATH = "artifacts/models/logreg.pkl"
SVM_PATH = "artifacts/models/svm.pkl"
TEXTCNN_PATH = "artifacts/models/textcnn.pt"
RNN_PATH = "artifacts/models/rnn.pt"
LSTM_PATH = "artifacts/models/lstm.pt"
DISTILBERT_DIR = "artifacts/models/distilbert"
MAX_LENGTH_NN = 32
MAX_LENGTH_HF = 128

SUPPORTED_MODELS = {"classical", "svm", "textcnn", "rnn", "lstm", "transformer"}


@dataclass
class PredictionResult:
    intent: str
    confidence: float
    top5: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    is_oos: bool = False
    model_used: str = ""


class Predictor:
    def __init__(self, model_type: str):
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type: {model_type}, must be one of {SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.label_map = load_label_map()
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vectorizer = None
        self._vocab = None
        self._model = None
        self._tokenizer = None

        self.load()

    def load(self) -> None:
        if self.model_type in ("classical", "svm"):
            self._vectorizer = load_vectorizer(VECTORIZER_PATH)
            if self.model_type == "classical":
                self._model = LogisticRegressionModel()
                self._model.load(LOGREG_PATH)
            else:
                self._model = SVMModel()
                self._model.load(SVM_PATH)

        elif self.model_type in ("textcnn", "rnn", "lstm"):
            self._vocab = Vocabulary.load(VOCAB_PATH)
            neural_config = load_config("neural")
            model_cfg = neural_config["model"][self.model_type]
            num_classes = len(self.label_map)

            if self.model_type == "textcnn":
                self._model = TextCNN(
                    vocab_size=len(self._vocab),
                    embedding_dim=model_cfg["embedding_dim"],
                    num_filters=model_cfg["num_filters"],
                    kernel_sizes=model_cfg["kernel_sizes"],
                    num_classes=num_classes,
                    dropout=model_cfg["dropout"],
                )
                self._model.load(TEXTCNN_PATH)
            elif self.model_type == "rnn":
                self._model = RNNModel(
                    vocab_size=len(self._vocab),
                    embedding_dim=model_cfg["embedding_dim"],
                    hidden_dim=model_cfg["hidden_dim"],
                    num_layers=model_cfg["num_layers"],
                    num_classes=num_classes,
                    dropout=model_cfg["dropout"],
                )
                self._model.load(RNN_PATH)
            else:
                self._model = LSTMModel(
                    vocab_size=len(self._vocab),
                    embedding_dim=model_cfg["embedding_dim"],
                    hidden_dim=model_cfg["hidden_dim"],
                    num_layers=model_cfg["num_layers"],
                    num_classes=num_classes,
                    dropout=model_cfg["dropout"],
                )
                self._model.load(LSTM_PATH)

            self._model.to(self.device)
            self._model.eval()

        elif self.model_type == "transformer":
            self._model = TransformerModel(
                model_name=DISTILBERT_DIR,
                num_labels=len(self.label_map),
            )
            self._model.model.to(self.device)
            self._model.model.eval()

    def _predict_proba(self, text: str) -> np.ndarray:
        cleaned = clean_text(text)

        if self.model_type in ("classical", "svm"):
            X = transform(self._vectorizer, [cleaned])
            if self.model_type == "classical":
                return self._model.predict_proba(X)[0]
            scores = self._model.model.decision_function(X)[0]
            exp_scores = np.exp(scores - np.max(scores))
            return exp_scores / exp_scores.sum()

        if self.model_type in ("textcnn", "rnn", "lstm"):
            encoded = self._vocab.encode(cleaned, MAX_LENGTH_NN)
            tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
            return self._model.predict_proba(tensor)[0]

        if self.model_type == "transformer":
            return self._model.predict_proba([cleaned], MAX_LENGTH_HF)[0]

        raise ValueError(f"unsupported model_type: {self.model_type}")

    def predict(self, text: str) -> PredictionResult:
        start = time.perf_counter()
        probs = self._predict_proba(text)
        latency_ms = (time.perf_counter() - start) * 1000

        top_indices = np.argsort(probs)[-5:][::-1]
        top5 = [
            {"intent": self.id_to_label[idx], "confidence": round(float(probs[idx]), 4)}
            for idx in top_indices
        ]

        best_idx = int(top_indices[0])
        intent = self.id_to_label[best_idx]
        confidence = float(probs[best_idx])
        is_oos = confidence < settings.oos_threshold

        return PredictionResult(
            intent=intent,
            confidence=round(confidence, 4),
            top5=top5,
            latency_ms=round(latency_ms, 3),
            is_oos=is_oos,
            model_used=self.model_type,
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        return [self.predict(text) for text in texts]
