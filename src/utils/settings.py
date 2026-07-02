from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = "intent-classifier-artifacts"

    mlflow_tracking_uri: str = "http://localhost:5000"

    model_type: str = "transformer"
    ab_model_a: str = "classical"
    ab_model_b: str = "transformer"
    ab_split: float = 0.3

    oos_threshold: float = 0.5


settings = Settings()
