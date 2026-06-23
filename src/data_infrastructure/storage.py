"""MinIO object storage client for dataset and artifact management."""

import io
import json
from pathlib import Path

from minio import Minio
from minio.error import S3Error


class StorageClient:
    BUCKETS = ["datasets", "models", "mlflow-artifacts", "checkpoints"]

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin123",
    ):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    def setup_buckets(self):
        for bucket in self.BUCKETS:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    def upload_file(self, bucket: str, object_name: str, file_path: str):
        self.client.fput_object(bucket, object_name, file_path)

    def upload_bytes(self, bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream"):
        self.client.put_object(
            bucket, object_name, io.BytesIO(data), len(data), content_type=content_type
        )

    def download_file(self, bucket: str, object_name: str, file_path: str):
        self.client.fget_object(bucket, object_name, file_path)

    def upload_dataset(self, dataset_name: str, version: str, local_path: str):
        """Upload a processed dataset directory to MinIO."""
        local_path = Path(local_path)
        prefix = f"{dataset_name}/{version}/"

        for file in local_path.rglob("*"):
            if file.is_file():
                object_name = prefix + str(file.relative_to(local_path))
                self.upload_file("datasets", object_name, str(file))

    def upload_checkpoint(self, run_id: str, checkpoint_path: str):
        """Upload a training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        prefix = f"{run_id}/"

        for file in checkpoint_path.rglob("*"):
            if file.is_file():
                object_name = prefix + str(file.relative_to(checkpoint_path))
                self.upload_file("checkpoints", object_name, str(file))

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def upload_metadata(self, bucket: str, key: str, metadata: dict):
        data = json.dumps(metadata, indent=2).encode()
        self.upload_bytes(bucket, f"{key}/metadata.json", data, "application/json")


def setup_storage():
    """Initialize storage buckets."""
    client = StorageClient()
    client.setup_buckets()
    print("Storage buckets created successfully")
    return client


if __name__ == "__main__":
    setup_storage()
