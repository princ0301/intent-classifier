from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from src.utils.settings import settings


def _get_client():
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )


def upload_artifact(local_path: str, s3_key: str) -> None:
    client = _get_client()
    client.upload_file(local_path, settings.s3_bucket, s3_key)
    print(f"uploaded {local_path} -> s3://{settings.s3_bucket}/{s3_key}")


def download_artifact(s3_key: str, local_path: str) -> None:
    client = _get_client()
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(settings.s3_bucket, s3_key, local_path)
    print(f"downloaded s3://{settings.s3_bucket}/{s3_key} -> {local_path}")


def artifact_exists(s3_key: str) -> bool:
    client = _get_client()
    try:
        client.head_object(Bucket=settings.s3_bucket, Key=s3_key)
        return True
    except ClientError:
        return False


def list_artifacts(prefix: str) -> list[str]:
    client = _get_client()
    response = client.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)
    if "Contents" not in response:
        return []
    return [obj["Key"] for obj in response["Contents"]]
