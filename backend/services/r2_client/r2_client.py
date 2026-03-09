"""Protocol for R2/S3 compatible object storage."""

from __future__ import annotations

from typing import Protocol


class R2Client(Protocol):
    def upload_file(
        self, *, local_path: str, remote_key: str, content_type: str,
    ) -> str:
        """Upload a local file. Returns the public URL."""
        ...

    def is_configured(self) -> bool:
        """Return True if R2 credentials are set."""
        ...
