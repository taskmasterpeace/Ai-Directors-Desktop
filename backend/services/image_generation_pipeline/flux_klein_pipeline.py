"""FLUX.2 Klein 9B Base image generation pipeline wrapper."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from diffusers import AutoencoderKL, BitsAndBytesConfig, Flux2KleinPipeline, PipelineQuantizationConfig  # type: ignore[reportUnknownVariableType]
from PIL import Image
from PIL.Image import Image as PILImage

from services.services_utils import ImagePipelineOutputLike, PILImageType, get_device_type

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _FluxKleinOutput:
    images: Sequence[PILImageType]


def _latents_to_pil(model_path: str, latents: torch.Tensor) -> list[PILImageType]:
    """Decode latents to PIL via a fresh VAE loaded from disk.

    The pipeline's built-in AutoencoderKLFlux2 segfaults on Windows/CUDA
    when accelerate model-cpu-offload hooks are active.  Loading a clean
    AutoencoderKL from the ``vae/`` subfolder and decoding on CPU in
    float32 is the only reliable workaround.
    """
    import pathlib

    latents_cpu = latents.to("cpu")

    gc.collect()
    torch.cuda.empty_cache()

    vae_path = str(pathlib.Path(model_path) / "vae")
    _logger.info("Loading fresh VAE from %s for decode", vae_path)
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32)  # type: ignore[reportUnknownMemberType]
    vae = vae.to("cpu")  # type: ignore[reportUnknownMemberType]
    vae.eval()  # type: ignore[reportUnknownMemberType]

    latents_f32 = latents_cpu.to(dtype=torch.float32)
    with torch.no_grad():
        decoded = vae.decode(latents_f32, return_dict=False)[0]  # type: ignore[reportUnknownMemberType]

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    images: list[PILImageType] = []
    for i in range(decoded.shape[0]):
        arr = decoded[i].permute(1, 2, 0).numpy()
        pil_img = Image.fromarray((arr * 255).astype(np.uint8))
        images.append(pil_img)

    del vae, latents_cpu, latents_f32, decoded
    gc.collect()

    return images


class FluxKleinImagePipeline:
    """FLUX.2 Klein 9B Base — text-to-image, img2img, and LoRA support.

    Loads the transformer with bitsandbytes NF4 quantization (~5GB instead
    of ~18GB bf16).  Still uses enable_model_cpu_offload() because the
    T5-XXL text encoder (~9GB bf16) plus the NF4 transformer exceeds 24GB
    VRAM when activation memory is included.

    After denoising, the pipeline is destroyed to release the accelerate
    hooks that cause Windows/CUDA VAE segfaults, then latents are decoded
    via a fresh VAE on CPU.

    The PipelinesHandler detects the destroyed pipeline and recreates it
    on the next generation request (~10s rebuild with NF4).
    """

    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "FluxKleinImagePipeline":
        return FluxKleinImagePipeline(model_path=model_path, device=device)

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self._device: str | None = None
        self._model_offload_active = False
        self._lora_loaded: str | None = None
        self._model_path = model_path

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        quant_config = PipelineQuantizationConfig(
            quant_mapping={"transformer": nf4_config},
        )

        self.pipeline = Flux2KleinPipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model_path,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        if device is not None:
            self.to(device)

    def _resolve_generator_device(self) -> str:
        if self._model_offload_active:
            return "cpu"
        if self._device is not None:
            return self._device
        execution_device = getattr(self.pipeline, "_execution_device", None)
        return get_device_type(execution_device)

    @staticmethod
    def _normalize_output(output: object) -> ImagePipelineOutputLike:
        images = getattr(output, "images", None)
        if not isinstance(images, Sequence):
            raise RuntimeError("Unexpected FLUX Klein pipeline output format: missing images sequence")

        images_list = cast(Sequence[object], images)
        validated_images: list[PILImageType] = []
        for image in images_list:
            if not isinstance(image, PILImage):
                raise RuntimeError("Unexpected FLUX Klein pipeline output: images must be PIL.Image instances")
            validated_images.append(image)

        return _FluxKleinOutput(images=validated_images)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        pipeline = cast(Any, self.pipeline)

        steps = num_inference_steps if num_inference_steps > 4 else 28
        gs = guidance_scale if guidance_scale > 0 else 4.0

        if self._model_offload_active:
            torch.cuda.empty_cache()
            output = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=gs,
                num_inference_steps=steps,
                generator=generator,
                output_type="latent",
                return_dict=True,
            )
            latents = output.images.to("cpu")
            # Destroy pipeline to release accelerate hooks before VAE decode.
            self._destroy_pipeline()
            pil_images = _latents_to_pil(self._model_path, latents)
            return _FluxKleinOutput(images=pil_images)

        output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=gs,
            num_inference_steps=steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
        return self._normalize_output(output)

    @torch.inference_mode()
    def img2img(
        self,
        prompt: str,
        image: PILImageType,
        strength: float,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        pipeline = cast(Any, self.pipeline)

        base_steps = num_inference_steps if num_inference_steps > 4 else 28
        effective_steps = max(1, int(base_steps * strength))
        gs = guidance_scale if guidance_scale > 0 else 4.0

        if self._model_offload_active:
            torch.cuda.empty_cache()
            output = pipeline(
                prompt=prompt,
                image=image,
                height=height,
                width=width,
                guidance_scale=gs,
                num_inference_steps=effective_steps,
                generator=generator,
                output_type="latent",
                return_dict=True,
            )
            latents = output.images.to("cpu")
            self._destroy_pipeline()
            pil_images = _latents_to_pil(self._model_path, latents)
            return _FluxKleinOutput(images=pil_images)

        output = pipeline(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            guidance_scale=gs,
            num_inference_steps=effective_steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
        return self._normalize_output(output)

    def _destroy_pipeline(self) -> None:
        """Destroy the diffusers pipeline to release accelerate hooks and VRAM.

        Required on Windows/CUDA: the accelerate model-cpu-offload hooks
        cause a segfault during VAE decode.  Destroying the pipeline before
        loading a fresh VAE avoids the crash entirely.

        After calling this, the pipeline object is gone.  The PipelinesHandler
        will detect this and recreate the pipeline on the next generation.
        """
        _logger.info("Destroying FLUX Klein pipeline to release accelerate hooks")
        del self.pipeline
        self._lora_loaded = None
        self._model_offload_active = False
        gc.collect()
        torch.cuda.empty_cache()

    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device in ("cuda", "mps"):
            # Model-level CPU offload: moves whole sub-models to GPU one at
            # a time.  NF4 transformer (~5GB) + T5-XXL text_encoder (~9GB)
            # + activations still exceeds 24GB VRAM at 1024x1024.
            self.pipeline.enable_model_cpu_offload()  # type: ignore[reportUnknownMemberType]
            self._model_offload_active = True
        else:
            self._model_offload_active = False
            self.pipeline.to(runtime_device)  # type: ignore[reportUnknownMemberType]
        self._device = runtime_device

    def load_lora(self, lora_path: str, weight: float = 1.0) -> None:
        if self._lora_loaded == lora_path:
            return
        if self._lora_loaded is not None:
            self.unload_lora()
        pipeline = cast(Any, self.pipeline)
        pipeline.load_lora_weights(lora_path, adapter_name="user_lora")
        pipeline.set_adapters(["user_lora"], adapter_weights=[weight])
        self._lora_loaded = lora_path

    def unload_lora(self) -> None:
        if self._lora_loaded is None:
            return
        pipeline = cast(Any, self.pipeline)
        pipeline.unload_lora_weights()
        self._lora_loaded = None
