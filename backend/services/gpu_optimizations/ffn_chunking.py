"""Chunked feedforward optimization for LTX transformer.

Splits FeedForward.forward along the sequence dimension (dim=1) to reduce
peak VRAM.  Output is mathematically identical to unchunked forward —
FeedForward is pointwise along the sequence dimension so chunking is lossless.

Reference: RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management (V3.1)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
import torch

logger = logging.getLogger(__name__)

_MIN_SEQ_PER_CHUNK = 100  # skip chunking for short sequences


def _make_chunked_forward(
    original_forward: Callable[[torch.Tensor], torch.Tensor],
    num_chunks: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a drop-in replacement for FeedForward.forward that chunks along dim=1."""

    def chunked_forward(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            return original_forward(x)

        seq_len = x.shape[1]
        if seq_len < num_chunks * _MIN_SEQ_PER_CHUNK:
            return original_forward(x)

        chunk_size = (seq_len + num_chunks - 1) // num_chunks
        outputs: list[torch.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            outputs.append(original_forward(x[:, start:end, :]))
        return torch.cat(outputs, dim=1)

    return chunked_forward


def patch_ffn_chunking(model: torch.nn.Module, num_chunks: int = 8) -> int:
    """Monkey-patch all FeedForward modules in *model* to use chunked forward.

    Returns the number of modules patched.
    """
    patched = 0
    named: list[tuple[str, torch.nn.Module]] = list(model.named_modules())  # pyright: ignore[reportUnknownArgumentType]
    for name, module in named:
        net = getattr(module, "net", None)
        if net is None:
            continue
        if not isinstance(net, torch.nn.Sequential):
            continue
        if not (name.endswith(".ff") or name.endswith(".audio_ff")):
            continue

        original: Callable[[torch.Tensor], torch.Tensor] = module.forward  # type: ignore[assignment]
        module.forward = _make_chunked_forward(original, num_chunks)  # type: ignore[assignment]
        patched += 1

    if patched:
        logger.info("FFN chunking: patched %d feedforward modules (chunks=%d)", patched, num_chunks)
    return patched
