import random
from dataclasses import dataclass

from src.serving.predictor import PredictionResult, Predictor
from src.utils.settings import settings


@dataclass
class ABStats:
    model_name: str
    request_count: int = 0
    total_latency_ms: float = 0.0
    oos_count: int = 0
    confidence_sum: float = 0.0

    def record(self, result: PredictionResult) -> None:
        self.request_count += 1
        self.total_latency_ms += result.latency_ms
        self.confidence_sum += result.confidence
        if result.is_oos:
            self.oos_count += 1

    def summary(self) -> dict:
        if self.request_count == 0:
            return {
                "model_name": self.model_name,
                "request_count": 0,
                "avg_latency_ms": 0.0,
                "oos_rate": 0.0,
                "avg_confidence": 0.0,
            }
        return {
            "model_name": self.model_name,
            "request_count": self.request_count,
            "avg_latency_ms": round(self.total_latency_ms / self.request_count, 3),
            "oos_rate": round(self.oos_count / self.request_count, 4),
            "avg_confidence": round(self.confidence_sum / self.request_count, 4),
        }


class ABRouter:
    def __init__(
        self,
        model_a: str | None = None,
        model_b: str | None = None,
        split: float | None = None,
    ):
        self.model_a_name = model_a or settings.ab_model_a
        self.model_b_name = model_b or settings.ab_model_b
        self.split = split if split is not None else settings.ab_split

        self.predictor_a = Predictor(self.model_a_name)
        self.predictor_b = Predictor(self.model_b_name)

        self.stats_a = ABStats(model_name=self.model_a_name)
        self.stats_b = ABStats(model_name=self.model_b_name)

    def route(self, text: str) -> tuple[PredictionResult, str]:
        if random.random() < self.split:
            result = self.predictor_b.predict(text)
            self.stats_b.record(result)
            return result, "B"
        result = self.predictor_a.predict(text)
        self.stats_a.record(result)
        return result, "A"

    def predict(self, text: str) -> dict:
        result, variant = self.route(text)
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "top5": result.top5,
            "latency_ms": result.latency_ms,
            "is_oos": result.is_oos,
            "model_used": result.model_used,
            "ab_variant": variant,
        }

    def get_stats(self) -> dict:
        return {
            "model_a": self.stats_a.summary(),
            "model_b": self.stats_b.summary(),
            "split": self.split,
        }

    def reset_stats(self) -> None:
        self.stats_a = ABStats(model_name=self.model_a_name)
        self.stats_b = ABStats(model_name=self.model_b_name)
