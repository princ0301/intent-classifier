from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    model_type: str | None = None


class TopIntent(BaseModel):
    intent: str
    confidence: float


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    top5: list[TopIntent]
    latency_ms: float
    is_oos: bool
    model_used: str
    ab_variant: str | None = None


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    model_type: str | None = None


class BatchResponse(BaseModel):
    predictions: list[PredictResponse]
    total_latency_ms: float


class ModelInfo(BaseModel):
    name: str
    alias: str
    version: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


class DriftResponse(BaseModel):
    drift_summary: dict
    confidence_drift: dict
    oos_rate_drift: dict


class ABStatsResponse(BaseModel):
    model_a: dict
    model_b: dict
    split: float
