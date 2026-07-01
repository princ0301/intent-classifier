FROM python:3.10-slim AS builder

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml .
COPY src/ src/
COPY api/ api/
COPY app/ app/

RUN uv pip install --system --no-cache -r pyproject.toml 2>/dev/null || \
    uv pip install --system --no-cache \
        torch==2.6.0 \
        transformers==4.51.3 \
        accelerate==1.5.2 \
        datasets==3.3.2 \
        scikit-learn==1.6.1 \
        numpy==2.2.3 \
        pandas==2.2.3 \
        mlflow==2.21.3 \
        fastapi==0.115.12 \
        uvicorn==0.34.0 \
        streamlit==1.43.2 \
        pyyaml==6.0.2 \
        python-dotenv==1.0.1 \
        boto3==1.37.38 \
        evidently==0.7.5 \
        httpx==0.28.1 \
        pydantic-settings==2.8.1 \
        matplotlib==3.10.1 \
        seaborn==0.13.2


FROM python:3.10-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY configs/ configs/

RUN mkdir -p artifacts/models artifacts/vectorizers artifacts/monitoring artifacts/evaluation mlruns data/raw

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]