from fastapi import APIRouter, HTTPException, Request

from api.schemas import BatchRequest, BatchResponse, PredictRequest, PredictResponse, TopIntent
from src.monitoring.drift import append_current_data

router = APIRouter()


def _to_predict_response(result, ab_variant: str | None = None) -> PredictResponse:
    return PredictResponse(
        intent=result.intent,
        confidence=result.confidence,
        top5=[TopIntent(**item) for item in result.top5],
        latency_ms=result.latency_ms,
        is_oos=result.is_oos,
        model_used=result.model_used,
        ab_variant=ab_variant,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    predictors = request.app.state.predictors
    ab_router = request.app.state.ab_router

    if payload.model_type:
        if payload.model_type not in predictors:
            raise HTTPException(status_code=400, detail=f"unknown model_type: {payload.model_type}")
        result = predictors[payload.model_type].predict(payload.text)
        response = _to_predict_response(result)
    else:
        result_dict = ab_router.predict(payload.text)
        response = PredictResponse(
            intent=result_dict["intent"],
            confidence=result_dict["confidence"],
            top5=[TopIntent(**item) for item in result_dict["top5"]],
            latency_ms=result_dict["latency_ms"],
            is_oos=result_dict["is_oos"],
            model_used=result_dict["model_used"],
            ab_variant=result_dict["ab_variant"],
        )

    append_current_data(
        texts=[payload.text],
        predictions=[response.intent],
        confidences=[response.confidence],
    )

    return response


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, request: Request) -> BatchResponse:
    predictors = request.app.state.predictors

    model_type = payload.model_type or "transformer"
    if model_type not in predictors:
        raise HTTPException(status_code=400, detail=f"unknown model_type: {model_type}")

    predictor = predictors[model_type]
    results = predictor.predict_batch(payload.texts)
    responses = [_to_predict_response(r) for r in results]

    total_latency = sum(r.latency_ms for r in responses)

    append_current_data(
        texts=payload.texts,
        predictions=[r.intent for r in responses],
        confidences=[r.confidence for r in responses],
    )

    return BatchResponse(predictions=responses, total_latency_ms=round(total_latency, 3))