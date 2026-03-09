# GPU Optimizations + R2 Storage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Dramatically improve GPU generation speed and VRAM efficiency, add R2 cloud upload, commit+push pending security work.

**Architecture:** Add a `gpu_optimizations` service module that monkey-patches the loaded LTX transformer at pipeline creation time. FFN chunking reduces peak VRAM by splitting feedforward along the sequence dimension. TeaCache wraps the denoising function to skip redundant transformer passes. R2 upload uses boto3 S3-compatible client post-generation. All optimizations are toggleable via AppSettings.

**Tech Stack:** PyTorch, ltx_core (monkey-patching), boto3 (R2/S3), numpy (TeaCache polynomial)

---

### Task 1: Commit + push security fixes and README

These changes already exist in the working tree from a previous session.

**Step 1: Review staged changes**

Run: `cd D:/git/directors-desktop && git status`

**Step 2: Commit security fixes**

Run:
```bash
git add backend/services/palette_sync_client/palette_sync_client_impl.py electron/main.ts electron/ipc/file-handlers.ts README.md
git commit -m "fix: externalize Supabase credentials, redact auth tokens in logs, update README with new features"
```

**Step 3: Push**

Run: `git push origin main`

---

### Task 2: Add FFN Chunked Feedforward optimization

Reduces peak VRAM by 8x in feedforward layers. Mathematically identical output.

**Files:**
- Create: `backend/services/gpu_optimizations/__init__.py`
- Create: `backend/services/gpu_optimizations/ffn_chunking.py`
- Test: `backend/tests/test_ffn_chunking.py`

**Step 1: Create the module**

Create `backend/services/gpu_optimizations/__init__.py` (empty file).

Create `backend/services/gpu_optimizations/ffn_chunking.py`:

```python
"""Chunked feedforward optimization for LTX transformer.

Splits FeedForward.forward along the sequence dimension (dim=1) to reduce
peak VRAM.  Output is mathematically identical to unchunked forward —
FeedForward is pointwise along the sequence dimension so chunking is lossless.

Reference: RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management (V3.1)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

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
    for name, module in model.named_modules():
        if not hasattr(module, "net"):
            continue
        if not isinstance(module.net, torch.nn.Sequential):
            continue
        # Match FeedForward modules (they live at .ff and .audio_ff in each block)
        if not (name.endswith(".ff") or name.endswith(".audio_ff")):
            continue

        original = module.forward
        module.forward = _make_chunked_forward(original, num_chunks)  # type: ignore[assignment]
        patched += 1

    if patched:
        logger.info("FFN chunking: patched %d feedforward modules (chunks=%d)", patched, num_chunks)
    return patched
```

**Step 2: Write test**

Create `backend/tests/test_ffn_chunking.py`:

```python
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
```

**Step 3: Run tests**

Run: `cd backend && uv run pytest tests/test_ffn_chunking.py -v --tb=short`
Expected: 4 tests PASS

**Step 4: Commit**

```bash
git add backend/services/gpu_optimizations/ backend/tests/test_ffn_chunking.py
git commit -m "feat: add FFN chunked feedforward to reduce peak VRAM by up to 8x"
```

---

### Task 3: Add TeaCache optimization

Caches transformer residuals between denoising steps to skip redundant computation. 1.6-2.1x speedup.

**Files:**
- Create: `backend/services/gpu_optimizations/tea_cache.py`
- Test: `backend/tests/test_tea_cache.py`

**Step 1: Create TeaCache module**

Create `backend/services/gpu_optimizations/tea_cache.py`:

```python
"""TeaCache: Timestep-Aware Caching for diffusion denoising loops.

Wraps a denoising function to skip transformer forward passes when the
timestep embedding hasn't changed significantly from the previous step.
First and last steps are always computed.

Reference: ali-vilab/TeaCache (TeaCache4LTX-Video)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Polynomial fitted to LTX-Video noise schedule for rescaling relative L1 distance
_RESCALE_COEFFICIENTS = [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03]
_rescale_poly = np.poly1d(_RESCALE_COEFFICIENTS)


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
```

**Step 2: Write test**

Create `backend/tests/test_tea_cache.py`:

```python
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


def _make_fake_denoise():
    call_count = [0]

    def denoise_fn(video_state, audio_state, sigmas, step_index):
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
```

**Step 3: Run tests**

Run: `cd backend && uv run pytest tests/test_tea_cache.py -v --tb=short`
Expected: 4 tests PASS

**Step 4: Commit**

```bash
git add backend/services/gpu_optimizations/tea_cache.py backend/tests/test_tea_cache.py
git commit -m "feat: add TeaCache timestep-aware caching for 1.6-2.1x speedup"
```

---

### Task 4: Integrate optimizations into pipeline lifecycle

Wire FFN chunking and TeaCache into the existing pipeline creation and generation flow.

**Files:**
- Modify: `backend/state/app_settings.py` (add settings)
- Modify: `backend/handlers/pipelines_handler.py` (apply FFN chunking at load time)
- Modify: `backend/services/ltx_pipeline_common.py` (wrap denoising loop with TeaCache)
- Modify: `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py` (accept tea_cache_threshold)

**Step 1: Add settings fields**

In `backend/state/app_settings.py`, add to `AppSettings` class after `batch_sound_enabled`:

```python
    ffn_chunk_count: int = 8
    tea_cache_threshold: float = 0.0
```

Add validator:
```python
    @field_validator("ffn_chunk_count", mode="before")
    @classmethod
    def _clamp_ffn_chunk_count(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=32, default=8)
```

Also add to `SettingsResponse`:
```python
    ffn_chunk_count: int = 8
    tea_cache_threshold: float = 0.0
```

**Step 2: Apply FFN chunking in PipelinesHandler**

In `backend/handlers/pipelines_handler.py`, add import at top:

```python
from services.gpu_optimizations.ffn_chunking import patch_ffn_chunking
```

In `_create_video_pipeline` method, after `self._compile_if_enabled(state)` on line 139, add FFN chunking:

```python
    def _create_video_pipeline(self, model_type: VideoPipelineModelType) -> VideoPipelineState:
        # ... existing code creating pipeline and state ...
        state = self._compile_if_enabled(state)

        # Apply FFN chunking if enabled and torch.compile is not active
        chunk_count = self.state.app_settings.ffn_chunk_count
        if chunk_count > 0 and not state.is_compiled:
            transformer = state.pipeline.pipeline.model_ledger.transformer()
            patch_ffn_chunking(transformer, num_chunks=chunk_count)

        return state
```

**Step 3: Wire TeaCache into DistilledNativePipeline**

In `backend/services/ltx_pipeline_common.py`, add import:

```python
from services.gpu_optimizations.tea_cache import wrap_denoise_fn_with_tea_cache
```

In `DistilledNativePipeline.__init__`, add a `tea_cache_threshold` parameter:

```python
    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device | None = None,
        fp8transformer: bool = False,
        tea_cache_threshold: float = 0.0,
    ) -> None:
        # ... existing init code ...
        self._tea_cache_threshold = tea_cache_threshold
```

In `DistilledNativePipeline.__call__`, wrap the denoising_loop function before it's called:

After the `denoising_loop` closure is defined (around line 147), add:

```python
        denoising_loop_fn = denoising_loop
        if self._tea_cache_threshold > 0:
            num_steps = len(sigmas) - 1
            cached_denoise = wrap_denoise_fn_with_tea_cache(
                simple_denoising_func(video_context=video_context, audio_context=audio_context, transformer=transformer),
                num_steps=num_steps,
                threshold=self._tea_cache_threshold,
            )

            def tea_cache_loop(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper: EulerDiffusionStep,
            ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=cached_denoise,
                )

            denoising_loop_fn = tea_cache_loop
```

Then use `denoising_loop_fn` instead of `denoising_loop` in the `denoise_audio_video` call.

**Step 4: Run existing tests**

Run: `cd backend && uv run pytest -v --tb=short`
Expected: All existing tests still pass (optimizations only activate with real GPU models)

**Step 5: Commit**

```bash
git add backend/state/app_settings.py backend/handlers/pipelines_handler.py backend/services/ltx_pipeline_common.py backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py
git commit -m "feat: wire FFN chunking and TeaCache into pipeline lifecycle"
```

---

### Task 5: Add VRAM cleanup after every generation

Prevents post-heavy-load GPU degradation (stalling at 15% after long generations).

**Files:**
- Modify: `backend/handlers/queue_worker.py` (add cleanup after job completion)
- Modify: `backend/services/gpu_cleaner/torch_cleaner.py` (add aggressive cleanup)

**Step 1: Add aggressive cleanup method**

In `backend/services/gpu_cleaner/torch_cleaner.py`:

```python
class TorchCleaner:
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
```

**Step 2: Add cleanup in QueueWorker after job completion**

In `backend/handlers/queue_worker.py`, the `_run_job` method calls `executor.execute(job)` in a try/finally. Add a `gpu_cleaner` parameter to QueueWorker and call cleanup after GPU jobs:

Add to `__init__`:
```python
    def __init__(
        self,
        *,
        queue: JobQueue,
        gpu_executor: JobExecutor,
        api_executor: JobExecutor,
        gpu_cleaner: GpuCleaner | None = None,
        on_batch_complete: Callable[[str, list[QueueJob]], None] | None = None,
        enhance_handler: EnhancePromptProvider | None = None,
    ) -> None:
        # ... existing ...
        self._gpu_cleaner = gpu_cleaner
```

In `_run_job`, add cleanup after GPU jobs complete:
```python
    def _run_job(self, job: QueueJob, executor: JobExecutor, slot: str) -> None:
        try:
            result_paths = executor.execute(job)
            self._queue.update_job(job.id, status="complete", progress=100, phase="complete", result_paths=result_paths)
        except Exception as exc:
            logger.error("Job %s failed: %s", job.id, exc)
            self._queue.update_job(job.id, status="error", error=str(exc))
        finally:
            if slot == "gpu" and self._gpu_cleaner is not None:
                try:
                    self._gpu_cleaner.deep_cleanup()
                except Exception:
                    pass
            with self._lock:
                if slot == "gpu":
                    self._gpu_busy = False
                else:
                    self._api_busy = False
```

**Step 3: Wire gpu_cleaner in AppHandler** (where QueueWorker is constructed)

Find where `QueueWorker` is instantiated in `backend/app_handler.py` and pass the existing `gpu_cleaner` service.

**Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_queue_worker.py -v --tb=short`
Expected: PASS (existing tests use fakes, new param is optional)

**Step 5: Commit**

```bash
git add backend/services/gpu_cleaner/torch_cleaner.py backend/handlers/queue_worker.py backend/app_handler.py
git commit -m "feat: aggressive VRAM cleanup after every GPU generation"
```

---

### Task 6: R2 Storage upload

Upload generated videos/images to Cloudflare R2 after generation.

**Files:**
- Create: `backend/services/r2_client/__init__.py`
- Create: `backend/services/r2_client/r2_client.py` (Protocol)
- Create: `backend/services/r2_client/r2_client_impl.py` (boto3 implementation)
- Modify: `backend/handlers/job_executors.py` (upload after completion)
- Modify: `backend/state/app_settings.py` (R2 settings)
- Test: `backend/tests/test_r2_upload.py`

**Step 1: Add boto3 dependency**

In `backend/pyproject.toml`, add to dependencies:
```toml
    "boto3>=1.34.0",
```

Run: `cd backend && uv sync --frozen --extra test --extra dev` (or `uv lock` if lock needs updating)

**Step 2: Add R2 settings**

In `backend/state/app_settings.py`, add to `AppSettings`:
```python
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_endpoint: str = ""
    r2_bucket: str = ""
    r2_public_url: str = ""
    auto_upload_to_r2: bool = False
```

Add to `SettingsResponse`:
```python
    has_r2_credentials: bool = False
    auto_upload_to_r2: bool = False
```

Update `to_settings_response` to handle R2 fields:
```python
    r2_key = data.pop("r2_access_key_id", "")
    data.pop("r2_secret_access_key", "")
    data.pop("r2_endpoint", "")
    data.pop("r2_bucket", "")
    data.pop("r2_public_url", "")
    data["has_r2_credentials"] = bool(r2_key)
```

**Step 3: Create R2 client protocol**

Create `backend/services/r2_client/__init__.py` (empty).

Create `backend/services/r2_client/r2_client.py`:

```python
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
```

**Step 4: Create R2 client implementation**

Create `backend/services/r2_client/r2_client_impl.py`:

```python
"""Cloudflare R2 client using boto3 S3-compatible API."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

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

        import boto3

        s3 = boto3.client(
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
```

**Step 5: Hook upload into job executors**

In `backend/handlers/job_executors.py`, add upload after GPU job completion.

Add to `GpuJobExecutor.__init__`:
```python
    def __init__(self, handler: AppHandler) -> None:
        self._handler = handler

    def _try_upload_to_r2(self, job: QueueJob, result_paths: list[str]) -> None:
        """Upload results to R2 if configured."""
        settings = self._handler.state.app_settings
        if not settings.auto_upload_to_r2:
            return
        if not (settings.r2_access_key_id and settings.r2_endpoint):
            return

        from services.r2_client.r2_client_impl import R2ClientImpl

        client = R2ClientImpl(
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            endpoint=settings.r2_endpoint,
            bucket=settings.r2_bucket,
            public_url=settings.r2_public_url,
        )

        for path in result_paths:
            try:
                ext = Path(path).suffix
                content_type = "video/mp4" if ext == ".mp4" else "image/png"
                remote_key = f"videos/{job.id}{ext}"
                client.upload_file(local_path=path, remote_key=remote_key, content_type=content_type)
            except Exception as exc:
                logger.warning("R2 upload failed for %s: %s", path, exc)
```

Add `from pathlib import Path` import and call `self._try_upload_to_r2(job, result)` after each execute in `GpuJobExecutor.execute()`:

```python
    def execute(self, job: QueueJob) -> list[str]:
        syncer = _ProgressSyncer(self._handler, job.id)
        syncer.start()
        try:
            if job.type == "image":
                result = self._execute_image(job)
            elif job.type == "video":
                result = self._execute_video(job)
            else:
                raise ValueError(f"Unknown job type: {job.type}")
            self._try_upload_to_r2(job, result)
            return result
        finally:
            syncer.stop()
```

**Step 6: Write test**

Create `backend/tests/test_r2_upload.py`:

```python
"""Tests for R2 upload integration."""

from __future__ import annotations


def test_r2_client_is_configured_when_credentials_present() -> None:
    from services.r2_client.r2_client_impl import R2ClientImpl

    client = R2ClientImpl(
        access_key_id="test",
        secret_access_key="test",
        endpoint="https://example.com",
        bucket="test-bucket",
        public_url="https://pub.example.com",
    )
    assert client.is_configured() is True


def test_r2_client_not_configured_when_empty() -> None:
    from services.r2_client.r2_client_impl import R2ClientImpl

    client = R2ClientImpl(
        access_key_id="",
        secret_access_key="",
        endpoint="",
        bucket="",
        public_url="",
    )
    assert client.is_configured() is False
```

**Step 7: Run tests**

Run: `cd backend && uv run pytest tests/test_r2_upload.py -v --tb=short`
Expected: PASS

**Step 8: Commit**

```bash
git add backend/services/r2_client/ backend/handlers/job_executors.py backend/state/app_settings.py backend/pyproject.toml backend/tests/test_r2_upload.py
git commit -m "feat: add R2 cloud storage upload for generated videos/images"
```

---

### Task 7: Run typecheck and full test suite

**Step 1: TypeScript typecheck**

Run: `cd D:/git/directors-desktop && pnpm typecheck:ts`
Expected: PASS

**Step 2: Python typecheck**

Run: `cd D:/git/directors-desktop && pnpm typecheck:py`
Expected: PASS (may need type: ignore for monkey-patches)

**Step 3: Full test suite**

Run: `cd backend && uv run pytest -v --tb=short`
Expected: All tests PASS

**Step 4: Fix any failures, commit fixes**

---

### Task 8: Benchmark with optimizations enabled

**Step 1: Start backend with optimizations**

Set `ffn_chunk_count: 8` and `tea_cache_threshold: 0.03` in settings.

**Step 2: Run 512p benchmark suite**

Test 512p at 2s, 5s, 8s, 10s. Compare against baseline:

| Test | Baseline | Target |
|------|----------|--------|
| 512p 2s | 37s | ~20-25s |
| 512p 5s | 84s | ~50-60s |
| 512p 8s | 100s | ~60-70s |
| 512p 10s | 651s | <200s (if FFN chunking fixes the cliff) |

**Step 3: Test 1080p 2s** (previously crashed with OOM)

If FFN chunking works, this should no longer OOM.

**Step 4: Document results in `docs/performance-report.md`**

**Step 5: Commit benchmark results**

---

### Task 9: SageAttention version bump (optional, if time permits)

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/install_sageattention.bat`

**Step 1: Bump version**

In `pyproject.toml`, change:
```
"sageattention>=1.0.0; sys_platform != 'darwin'",
```
to:
```
"sageattention>=2.0.0; sys_platform != 'darwin'",
```

**Step 2: Update install script**

Update `install_sageattention.bat` to clone the latest v2 branch.

**Step 3: Test that sageattention imports correctly**

Run: `cd backend && uv run python -c "import sageattention; print(sageattention.__version__)"`

**Step 4: Commit**

```bash
git add backend/pyproject.toml backend/install_sageattention.bat
git commit -m "chore: bump sageattention to v2 for improved attention performance"
```
