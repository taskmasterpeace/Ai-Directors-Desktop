"""Cloudflare R2 client using boto3 S3-compatible API."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class R2ClientImpl:
    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        endpoint: str,
        bucket: str,
        public_url: str,
    ) -> None:
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._endpoint = endpoint
        self._bucket = bucket
        self._public_url = public_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self._access_key_id and self._secret_access_key and self._endpoint and self._bucket)

    def upload_file(self, *, local_path: str, remote_key: str, content_type: str) -> str:
        if not self.is_configured():
            raise RuntimeError("R2 credentials not configured")

        boto3: Any = __import__("boto3")

        s3: Any = boto3.client(
            "s3",
            endpoint_url=self._endpoint,
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
        )

        content_type = content_type or mimetypes.guess_type(local_path)[0] or "application/octet-stream"
        s3.upload_file(
            local_path,
            self._bucket,
            remote_key,
            ExtraArgs={"ContentType": content_type},
        )

        public_url = f"{self._public_url}/{remote_key}"
        logger.info("Uploaded %s -> %s", Path(local_path).name, public_url)
        return public_url
