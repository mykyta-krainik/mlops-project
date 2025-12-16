import io
import os
from pathlib import Path
from typing import BinaryIO, Optional, Union

import pandas as pd
from minio import Minio
from minio.error import S3Error

from src.config import config


class MinioStorage:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: Optional[bool] = None,
    ):
        self.endpoint = endpoint or config.minio.endpoint
        self.access_key = access_key or config.minio.access_key
        self.secret_key = secret_key or config.minio.secret_key
        self.secure = secure if secure is not None else config.minio.secure

        self._client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    def ensure_bucket_exists(self, bucket_name: str) -> None:
        if not self._client.bucket_exists(bucket_name):
            self._client.make_bucket(bucket_name)

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Union[str, Path],
        content_type: str = "application/octet-stream",
    ) -> None:
        self.ensure_bucket_exists(bucket_name)
        file_path = Path(file_path)

        self._client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=str(file_path),
            content_type=content_type,
        )

    def upload_bytes(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.ensure_bucket_exists(bucket_name)

        data_stream = io.BytesIO(data)
        self._client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=data_stream,
            length=len(data),
            content_type=content_type,
        )

    def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Union[str, Path],
    ) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self._client.fget_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=str(file_path),
        )

    def download_bytes(self, bucket_name: str, object_name: str) -> bytes:
        response = self._client.get_object(bucket_name, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def upload_dataframe(
        self,
        bucket_name: str,
        object_name: str,
        df: pd.DataFrame,
        file_format: str = "csv",
    ) -> None:
        buffer = io.BytesIO()

        if file_format == "csv":
            df.to_csv(buffer, index=False)
            content_type = "text/csv"
        elif file_format == "parquet":
            df.to_parquet(buffer, index=False)
            content_type = "application/octet-stream"
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        buffer.seek(0)
        self.upload_bytes(bucket_name, object_name, buffer.getvalue(), content_type)

    def download_dataframe(
        self,
        bucket_name: str,
        object_name: str,
        file_format: str = "csv",
    ) -> pd.DataFrame:
        data = self.download_bytes(bucket_name, object_name)
        buffer = io.BytesIO(data)

        if file_format == "csv":
            return pd.read_csv(buffer)
        elif file_format == "parquet":
            return pd.read_parquet(buffer)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    def list_objects(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        recursive: bool = True,
    ) -> list[str]:
        objects = self._client.list_objects(
            bucket_name,
            prefix=prefix,
            recursive=recursive,
        )
        return [obj.object_name for obj in objects]

    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        try:
            self._client.stat_object(bucket_name, object_name)
            return True
        except S3Error:
            return False

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        self._client.remove_object(bucket_name, object_name)

    def get_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires_hours: int = 1,
    ) -> str:
        from datetime import timedelta

        return self._client.presigned_get_object(
            bucket_name,
            object_name,
            expires=timedelta(hours=expires_hours),
        )

