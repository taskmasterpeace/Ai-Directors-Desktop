"""TeaCache: Timestep-Aware Caching for diffusion denoising loops.

Wraps a denoising function to skip transformer forward passes when the
timestep embedding hasn't changed significantly from the previous step.
First and last steps are always computed.

Reference: ali-vilab/TeaCache (TeaCache4LTX-Video)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Polynomial fitted to LTX-Video noise schedule for rescaling relative L1 distance
_RESCALE_COEFFICIENTS = [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03]
_rescale_poly = np.poly1d(_RESCALE_COEFFICIENTS)

_original_euler_loop: Any = None


@dataclass
class TeaCacheState:
    """Mutable state held across denoising steps."""
    accumulated_distance: float = 0.0
    previous_residual: torch.Tensor | None = None
    previous_modulated_input: torch.Tensor | None = None
    step_count: int = 0
    skipped: int = 0
    computed: int = 0


def wrap_denoise_fn_with_tea_cache(
    denoise_fn: Any,
    num_steps: int,
    threshold: float,
) -> Any:
    """Wrap a denoising function with TeaCache.

    The wrapped function has the same signature as the original:
        denoise_fn(video_state, audio_state, sigmas, step_index)
            -> (denoised_video, denoised_audio)

    When the relative L1 distance of the timestep-modulated input is below
    *threshold*, the previous residual is reused instead of calling the
    transformer.

    Args:
        denoise_fn: Original denoising function.
        num_steps: Total number of denoising steps (len(sigmas) - 1).
        threshold: Caching threshold. 0 disables. 0.03 = balanced. 0.05 = aggressive.
    """
    if threshold <= 0:
        return denoise_fn

    state = TeaCacheState()

    def cached_denoise(
        video_state: Any,
        audio_state: Any,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Always compute first and last steps
        if step_index == 0 or step_index == num_steps - 1:
            should_compute = True
            state.accumulated_distance = 0.0
        elif state.previous_modulated_input is not None:
            # Estimate change using video_state latent as proxy for modulated input
            current = video_state.latent
            prev = state.previous_modulated_input
            rel_l1 = ((current - prev).abs().mean() / prev.abs().mean().clamp(min=1e-8)).item()
            rescaled = float(_rescale_poly(rel_l1))
            state.accumulated_distance += rescaled

            if state.accumulated_distance < threshold:
                should_compute = False
            else:
                should_compute = True
                state.accumulated_distance = 0.0
        else:
            should_compute = True
            state.accumulated_distance = 0.0

        state.previous_modulated_input = video_state.latent.clone()
        state.step_count += 1

        if not should_compute and state.previous_residual is not None:
            # Reuse cached residual
            cached_video = video_state.latent + state.previous_residual
            state.skipped += 1
            return cached_video, audio_state.latent
        else:
            # Full computation
            original_latent = video_state.latent.clone()
            denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_index)
            state.previous_residual = denoised_video - original_latent
            state.computed += 1
            return denoised_video, denoised_audio

    cached_denoise._tea_cache_state = state  # type: ignore[attr-defined]
    return cached_denoise


def install_tea_cache_patch(threshold: float) -> None:
    """Monkey-patch euler_denoising_loop in ltx_pipelines to apply TeaCache.

    This patches the function at the module level so that DistilledPipeline
    (which imports euler_denoising_loop inside __call__) picks it up
    automatically on each generation.
    """
    global _original_euler_loop

    import ltx_pipelines.utils.samplers as samplers_mod

    if _original_euler_loop is None:
        _original_euler_loop = samplers_mod.euler_denoising_loop

    if threshold <= 0:
        # Restore original
        samplers_mod.euler_denoising_loop = _original_euler_loop
        logger.info("TeaCache disabled — restored original euler_denoising_loop")
        return

    original = _original_euler_loop

    def tea_cache_euler_loop(
        sigmas: torch.Tensor,
        video_state: Any,
        audio_state: Any,
        stepper: Any,
        denoise_fn: Any,
        **kwargs: Any,
    ) -> Any:
        num_steps = len(sigmas) - 1
        cached_fn = wrap_denoise_fn_with_tea_cache(denoise_fn, num_steps=num_steps, threshold=threshold)
        result = original(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=cached_fn,
            **kwargs,
        )
        if hasattr(cached_fn, "_tea_cache_state"):
            s = cached_fn._tea_cache_state
            logger.info("TeaCache: computed %d, skipped %d of %d steps", s.computed, s.skipped, s.step_count)
        return result

    samplers_mod.euler_denoising_loop = tea_cache_euler_loop  # type: ignore[assignment]
    logger.info("TeaCache installed (threshold=%.3f)", threshold)


def uninstall_tea_cache_patch() -> None:
    """Restore the original euler_denoising_loop."""
    install_tea_cache_patch(0.0)
