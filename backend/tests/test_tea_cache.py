"""Tests for TeaCache denoising loop caching."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from services.gpu_optimizations.tea_cache import (
    TeaCacheState,
    wrap_denoise_fn_with_tea_cache,
)


@dataclass
class FakeLatentState:
    latent: torch.Tensor
    denoise_mask: torch.Tensor
    clean_latent: torch.Tensor


def _make_fake_denoise():  # type: ignore[no-untyped-def]
    call_count = [0]

    def denoise_fn(video_state, audio_state, sigmas, step_index):  # type: ignore[no-untyped-def]
        call_count[0] += 1
        return video_state.latent * 0.9, audio_state.latent * 0.9

    return denoise_fn, call_count


def test_tea_cache_disabled_when_threshold_zero() -> None:
    original, call_count = _make_fake_denoise()
    wrapped = wrap_denoise_fn_with_tea_cache(original, num_steps=10, threshold=0.0)
    assert wrapped is original  # no wrapping


def test_tea_cache_always_computes_first_step() -> None:
    original, call_count = _make_fake_denoise()
    wrapped = wrap_denoise_fn_with_tea_cache(original, num_steps=10, threshold=0.05)
    latent = torch.randn(1, 100, 64)
    vs = FakeLatentState(latent=latent, denoise_mask=torch.ones(1), clean_latent=latent)
    _v, _a = wrapped(vs, vs, torch.linspace(1, 0, 11), 0)
    assert call_count[0] == 1


def test_tea_cache_always_computes_last_step() -> None:
    original, call_count = _make_fake_denoise()
    wrapped = wrap_denoise_fn_with_tea_cache(original, num_steps=10, threshold=0.05)
    latent = torch.randn(1, 100, 64)
    vs = FakeLatentState(latent=latent, denoise_mask=torch.ones(1), clean_latent=latent)
    _v, _a = wrapped(vs, vs, torch.linspace(1, 0, 11), 0)
    _v, _a = wrapped(vs, vs, torch.linspace(1, 0, 11), 9)
    assert call_count[0] == 2  # both first and last always computed


def test_tea_cache_skips_similar_steps() -> None:
    original, call_count = _make_fake_denoise()
    wrapped = wrap_denoise_fn_with_tea_cache(original, num_steps=10, threshold=100.0)
    latent = torch.ones(1, 100, 64)
    vs = FakeLatentState(latent=latent, denoise_mask=torch.ones(1), clean_latent=latent)
    # Step 0: always computed
    wrapped(vs, vs, torch.linspace(1, 0, 11), 0)
    # Step 1: very high threshold means this should be skipped
    wrapped(vs, vs, torch.linspace(1, 0, 11), 1)
    tea_state: TeaCacheState = wrapped._tea_cache_state
    assert tea_state.skipped >= 1
