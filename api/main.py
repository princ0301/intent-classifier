from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routes import models, monitoring, predict
from api.schemas import HealthResponse
from src.serving.ab_router import ABRouter
from src.serving.predictor import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading predictors...")
    app.state.predictors = {
        "classical": Predictor("classical"),
        "svm": Predictor("svm"),
        "transformer": Predictor("transformer"),
    }
    print("loading AB router...")
    app.state.ab_router = ABRouter()
    print("startup complete")

    yield

    app.state.predictors.clear()


app = FastAPI(
    title="Intent Classifier API",
    description="End-to-end intent classification with classical ML, neural networks, and fine-tuned transformers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, tags=["predict"])
app.include_router(models.router, tags=["models"])
app.include_router(monitoring.router, tags=["monitoring"])


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=list(request.app.state.predictors.keys()),
    )
