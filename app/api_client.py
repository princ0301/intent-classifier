import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_predict(text: str, model_type: str | None = None) -> dict:
    payload = {"text": text}
    if model_type:
        payload["model_type"] = model_type
    response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def api_health() -> dict:
    response = requests.get(f"{API_BASE_URL}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def api_models() -> dict:
    response = requests.get(f"{API_BASE_URL}/models", timeout=10)
    response.raise_for_status()
    return response.json()


def api_ab_stats() -> dict:
    response = requests.get(f"{API_BASE_URL}/monitoring/ab-stats", timeout=10)
    response.raise_for_status()
    return response.json()


def api_drift() -> dict:
    response = requests.get(f"{API_BASE_URL}/monitoring/drift", timeout=30)
    response.raise_for_status()
    return response.json()


def api_reset_monitoring() -> dict:
    response = requests.post(f"{API_BASE_URL}/monitoring/reset", timeout=10)
    response.raise_for_status()
    return response.json()


def check_api_connection() -> bool:
    try:
        api_health()
        return True
    except requests.exceptions.RequestException:
        return False


def require_api():
    if not check_api_connection():
        st.error(
            f"Cannot connect to API at {API_BASE_URL}. "
            "Start it with: `uv run uvicorn api.main:app --reload --port 8000`"
        )
        st.stop()
