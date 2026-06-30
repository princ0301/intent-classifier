import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_loaded_models(client):
    response = client.get("/health")
    data = response.json()
    assert "models_loaded" in data
    assert len(data["models_loaded"]) > 0


def test_predict_returns_200(client):
    response = client.post("/predict", json={"text": "what is my account balance"})
    assert response.status_code == 200


def test_predict_returns_correct_schema(client):
    response = client.post("/predict", json={"text": "book a flight to paris"})
    data = response.json()
    assert "intent" in data
    assert "confidence" in data
    assert "top5" in data
    assert "is_oos" in data
    assert "model_used" in data


def test_predict_empty_text_returns_422(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_with_specific_model_type(client):
    response = client.post(
        "/predict", json={"text": "set an alarm", "model_type": "classical"}
    )
    data = response.json()
    assert data["model_used"] == "classical"


def test_predict_invalid_model_type_returns_400(client):
    response = client.post(
        "/predict", json={"text": "hello", "model_type": "not_a_model"}
    )
    assert response.status_code == 400


def test_predict_batch_returns_200(client):
    response = client.post(
        "/predict/batch", json={"texts": ["hello", "what time is it"]}
    )
    assert response.status_code == 200


def test_predict_batch_returns_correct_count(client):
    texts = ["hello", "goodbye", "what is the weather"]
    response = client.post("/predict/batch", json={"texts": texts})
    data = response.json()
    assert len(data["predictions"]) == len(texts)


def test_models_endpoint_returns_list(client):
    response = client.get("/models")
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 3


def test_ab_stats_endpoint(client):
    client.post("/predict", json={"text": "transfer money"})
    response = client.get("/monitoring/ab-stats")
    assert response.status_code == 200
    data = response.json()
    assert "model_a" in data
    assert "model_b" in data