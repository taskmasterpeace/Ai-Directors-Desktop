"""Tests for FFN chunked feedforward optimization."""

from __future__ import annotations

import torch

from services.gpu_optimizations.ffn_chunking import _make_chunked_forward, patch_ffn_chunking


class _FakeFeedForward(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FakeTransformerBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ff = _FakeFeedForward(dim)
        self.audio_ff = _FakeFeedForward(dim)


class _FakeTransformer(torch.nn.Module):
    def __init__(self, dim: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FakeTransformerBlock(dim) for _ in range(num_blocks)])


def test_chunked_forward_matches_original() -> None:
    dim = 32
    ff = _FakeFeedForward(dim)
    x = torch.randn(1, 2000, dim)

    original_out = ff(x)
    chunked_fn = _make_chunked_forward(ff.forward, num_chunks=4)
    chunked_out = chunked_fn(x)

    assert torch.allclose(original_out, chunked_out, atol=1e-5)


def test_chunked_forward_skips_short_sequences() -> None:
    dim = 16
    ff = _FakeFeedForward(dim)
    x = torch.randn(1, 50, dim)  # too short to chunk

    original_out = ff(x)
    chunked_fn = _make_chunked_forward(ff.forward, num_chunks=4)
    chunked_out = chunked_fn(x)

    assert torch.allclose(original_out, chunked_out, atol=1e-5)


def test_patch_ffn_chunking_patches_correct_modules() -> None:
    model = _FakeTransformer(dim=16, num_blocks=3)
    count = patch_ffn_chunking(model, num_chunks=4)
    assert count == 6  # 3 blocks x 2 (ff + audio_ff)


def test_patch_ffn_chunking_zero_when_no_match() -> None:
    model = torch.nn.Linear(16, 16)
    count = patch_ffn_chunking(model, num_chunks=4)
    assert count == 0
