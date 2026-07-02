from fastapi import APIRouter, Request

from api.schemas import ModelInfo, ModelsResponse

router = APIRouter()

REGISTERED_MODELS = [
    {"name": "intent-classifier-logreg", "alias": "challenger"},
    {"name": "intent-classifier-svm", "alias": "challenger"},
    {"name": "intent-classifier-distilbert", "alias": "champion"},
]


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(models=[ModelInfo(name=m["name"], alias=m["alias"]) for m in REGISTERED_MODELS])


@router.get("/models/loaded")
def list_loaded_models(request: Request) -> dict:
    predictors = request.app.state.predictors
    return {"loaded_models": list(predictors.keys())}
