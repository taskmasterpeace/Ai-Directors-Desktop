"""GPU cleaner service protocol definitions."""

from __future__ import annotations

from typing import Protocol


class GpuCleaner(Protocol):
    def cleanup(self) -> None:
        ...

    def deep_cleanup(self) -> None:
        ...
