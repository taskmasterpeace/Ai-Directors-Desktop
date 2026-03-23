"""NF4 (4-bit BitsAndBytes) quantized LTX video pipeline.

Uses BitsAndBytes NF4 quantization to load the LTX transformer at 4-bit
precision, following the same pattern as FluxKleinImagePipeline.
"""

from __future__ import annotations

import logging
from typing import Final

import torch

from api_types import ImageConditioningInput

logger = logging.getLogger(__name__)


class NF4FastVideoPipeline:
    """FastVideoPipeline implementation for NF4 quantized models.

    Scaffold only — raises NotImplementedError until tested with real models.
    """

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> "NF4FastVideoPipeline":
        return NF4FastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            lora_path=lora_path,
            lora_weight=lora_weight,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        lora_path: str | None = None,
        lora_weight: float = 1.0,
    ) -> None:
        raise NotImplementedError(
            "NF4 pipeline loading is not yet fully implemented. "
            "This requires testing with real NF4 quantized model files."
        )

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        raise NotImplementedError

    def warmup(self, output_path: str) -> None:
        raise NotImplementedError

    def compile_transformer(self) -> None:
        logger.info("Skipping torch.compile for NF4 pipeline — not supported with quantized weights")
