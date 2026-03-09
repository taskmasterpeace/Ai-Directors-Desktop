"""GPU cleanup helper service."""

from __future__ import annotations

import gc

import torch

from services.services_utils import empty_device_cache


class TorchCleaner:
    """Wraps GPU memory cleanup operations."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self._device = device

    def cleanup(self) -> None:
        empty_device_cache(self._device)
        gc.collect()

    def deep_cleanup(self) -> None:
        """Aggressive cleanup for after heavy GPU workloads."""
        gc.collect()
        empty_device_cache(self._device)
        gc.collect()
        empty_device_cache(self._device)
        if str(self._device) != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
