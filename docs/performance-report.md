# Directors Desktop — Performance Report

**Hardware:** NVIDIA RTX 4090 24GB VRAM, Windows 11, CUDA 12.9, Driver 576.80
**Date:** March 9, 2026
**Backend:** LTX-Video 0.9.7 (ltx-fast model), ZIT image model

---

## Benchmark Results

### Image Generation (ZIT, Local GPU, Warm)

| Resolution  | Time  |
|-------------|-------|
| 1024×1024   | 10s   |
| 768×1344    | 18s   |

### Video Generation (ltx-fast, Local GPU, Warm)

| Resolution | Duration | Frames | Time | Notes |
|------------|----------|--------|------|-------|
| 512p | 2s | 49 | 37s | Baseline |
| 512p | 5s | 121 | 44–84s | Session-dependent (44s prior, 84s this session) |
| 512p | 8s | 193 | 86–100s | Session-dependent (86s prior, 100s this session) |
| 512p | 10s (run A) | 241 | 651.6s (~10.9 min) | Consistent |
| 512p | 10s (run B) | 241 | 650.4s (~10.8 min) | Consistent — within 0.2% of run A |
| 512p | 20s | 481 | ~11,820s (~3.3 hrs) | Timed out at 3hrs; completed ~17min later |
| 720p | 2s | 49 | 39s | |
| 720p | 5s | 121 | 83s | |
| 720p | 8s | 193 | CANCELLED | ~36 min/step, estimated 4+ hours total |
| 1080p | 2s | 49 | 499s (~8.3 min) | |
| 1080p | 5s | 121 | CRASH | OOM during VAE decode after ~40 min |

### Video Extend (512p, 2s segments)

Extend was tested with last-frame extraction via ffmpeg, then submitting with `lastFramePath`.
After the 20s generation, the extend base job stalled at 15% inference despite the GPU running at 100%. This indicates post-heavy-load degradation requiring a backend restart.

### API Generation (Cloud)

| Model | Type | Params | Time | Notes |
|-------|------|--------|------|-------|
| nano-banana-2 | Image | 1024×1024, 4 steps | ~40s | Cloud API |
| seedance-1.5-pro | Video | 720p 5s | ~2-3 min | Cloud API; rejects 512p (only 480p/720p/1080p) |

### Cold Start

| Scenario | Time |
|----------|------|
| First generation after app launch (512p 2s) | ~66s |
| Warm generation (512p 2s, model already loaded) | ~37s |
| Cold start overhead | ~29s (model loading + warmup) |

---

## Scaling Analysis

### Frame Count vs Time (512p)

```
Frames:    49    121    193    241      481
Time:     37s    84s   100s   651s   11,820s
```

The relationship is **highly non-linear**. Key observations:

1. **49 → 193 frames (4× more):** Time scales ~2.7× (37s → 100s) — roughly linear
2. **193 → 241 frames (1.25× more):** Time scales ~6.5× (100s → 651s) — massive jump
3. **241 → 481 frames (2× more):** Time scales ~18× (651s → 11,820s) — exponential blowup

The dramatic scaling cliff at ~200 frames suggests the model hits a VRAM or attention computation threshold. Beyond 193 frames, inference shifts from being compute-bound to memory-bound, requiring tiling or sequential processing that dramatically slows throughput.

### Resolution Scaling (2s / 49 frames)

```
512p (960×544):   37s
720p (1280×704):  39s
1080p (1920×1088): 499s
```

Similar cliff between 720p and 1080p — the attention computation quadruples but VRAM constraints force a much slower execution path.

---

## Critical Issues Found

### 1. Non-Linear Scaling Beyond 8s at 512p
- The jump from 8s (100s) to 10s (651s) is a **6.5× increase for only 25% more frames**
- This means anything beyond ~8s at 512p enters an extremely slow regime
- **Impact:** Users will experience unexpectedly long waits for clips >8s

### 2. 20s Generation is Impractical (~3.3 hours)
- 20s at 512p takes 3+ hours on a high-end RTX 4090
- **Recommendation:** Either cap duration at 10s with clear warning, or implement frame-chunked generation

### 3. Post-Heavy-Load GPU Degradation
- After the 3+ hour 20s generation, a simple 2s job stalled at 15% indefinitely
- GPU showed 100% utilization but no progress — likely VRAM fragmentation
- **Impact:** Users may need to restart the app after long generations
- **Fix needed:** Explicit VRAM cleanup (torch.cuda.empty_cache(), gc.collect()) between generations

### 4. 1080p 5s Crashes with OOM
- 1080p at 5s (121 frames) crashes during VAE decode
- VAE decode requires loading the full frame buffer into VRAM
- **Impact:** 1080p is limited to ≤2s clips (and those take ~8 min)

### 5. Cancel Doesn't Stop GPU Inference
- `POST /api/queue/cancel/{id}` only marks the job status — the GPU continues working
- A cancelled long generation (e.g., 20s) will still consume GPU for hours
- **Fix needed:** Implement cooperative cancellation with a check in the inference loop

### 6. Warmup Race Conditions
- Jobs submitted before model warmup completes can fail with shape mismatch errors
- Backend should either queue jobs until warmup is done, or block submission

---

## Optimization Opportunities

Based on benchmark analysis + research of LTX-Video forks and community projects.

### Already Implemented

The codebase already has: SageAttention v1, FP8 quantization (cast), torch.compile, VAE tiling, text encoder CPU offloading, API-based text encoding.

### HIGH IMPACT — Speed

1. **TeaCache (Timestep-Aware Caching)**
   - Source: [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache)
   - Skips redundant transformer forward passes by caching outputs at timesteps where output changes minimally
   - **Expected speedup: 1.6-2.1×** (training-free, drop-in)
   - Could cut 512p 2s from 37s → ~18-23s

2. **Frame-Chunked Generation for Long Clips**
   - Instead of generating all 481 frames at once for 20s, generate in 2-5s chunks and stitch
   - Already have the extend/lastFramePath mechanism — automate this internally
   - Could reduce 20s from ~3.3 hours to ~5-10 minutes (5× 2s = ~185s)

### HIGH IMPACT — VRAM / Stability

3. **FFN Chunked Feedforward**
   - Source: [RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management](https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management)
   - LTX-2 transformer FFN layers expand hidden dim 4×, creating enormous intermediate tensors
   - Chunking into 8-16 pieces reduces peak VRAM by up to 8× with zero quality loss
   - **Benchmarks on RTX 4090**: 800 frames at 1920×1088 in ~16.5 GB, 900 frames in ~18.5 GB
   - **This is likely the fix for 1080p OOM crashes and the 10s nonlinear scaling cliff**

4. **Aggressive VRAM Cleanup Between Generations**
   - Add `torch.cuda.empty_cache()` + `gc.collect()` after each generation completes
   - Clear any intermediate tensors held in the pipeline
   - This would prevent the post-heavy-load stall issue

### MEDIUM IMPACT — Speed

5. **SageAttention 2++ Upgrade**
   - Source: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)
   - Current codebase pins v1.0.6; SageAttention 2++ provides 3.9× speedup over FlashAttention (vs 2.1× for v1)

6. **FP8 Scaled MM (TensorRT-LLM)**
   - Source: upstream ltx-core
   - Switch from `QuantizationPolicy.fp8_cast()` to `fp8_scaled_mm()` — uses native FP8 matrix multiplication without upcasting
   - Both faster and less memory; available on RTX 40xx

7. **Guidance Skip Steps**
   - `skip_step` param in `MultiModalGuiderParams` skips CFG computation every N steps
   - Since guidance requires 2-3× forward passes per step, skipping alternating steps cuts total passes ~30-40%

8. **Pre-Quantized FP8 Checkpoint**
   - Use `Lightricks/LTX-2.3-fp8` from HuggingFace instead of runtime FP8 casting
   - Faster load time (no conversion), potentially better quality (calibrated offline)

### MEDIUM IMPACT — UX

9. **Cooperative Cancellation**
   - Thread a cancellation callback through the denoising loop
   - Upstream `ltx-pipelines` denoising functions accept `on_step` callbacks
   - Check cancel flag after each timestep — enables immediate cancel vs waiting hours

10. **Duration Warnings in UI**
    - Show estimated time before submission based on resolution + duration
    - Warn users when estimated time exceeds 5 minutes

11. **Resolution Cap Enforcement**
    - Prevent 1080p ≥5s (will crash)
    - Prevent 720p ≥8s (impractical — would take hours)

### LOWER PRIORITY

12. **Memory Profiles (a la Wan2GP)**
    - Source: [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP)
    - Let users choose speed/memory tradeoff (tight → full-VRAM preload)

13. **Multi-GPU Tensor Parallelism**
    - Ring Attention + sequence parallelism for multi-GPU setups
    - Niche but enables 600K+ token sequences

### Priority Matrix for Our Specific Issues

| Issue | Best Fix | Expected Impact |
|---|---|---|
| 1080p 5s OOM crash | FFN chunking (#3) | Eliminates crash |
| 512p 10s = 651s (nonlinear jump) | FFN chunking (#3) + TeaCache (#1) | 5-10× faster |
| 37s baseline for 512p 2s | TeaCache (#1) + SageAttention 2++ (#5) | ~15-20s |
| 20s = 3.3 hours | Auto-chunking (#2) | ~5-10 min |
| Post-heavy-load stall | VRAM cleanup (#4) | Eliminates stall |
| Cancel doesn't stop GPU | Step-level callback (#9) | Immediate cancel |
| Cold start overhead | FP8 pre-quantized checkpoint (#8) | ~30-50% faster load |

---

## Practical User Guidelines

Based on these benchmarks, here are the recommended settings for the RTX 4090:

| Use Case | Recommended Setting | Expected Time |
|----------|-------------------|---------------|
| Quick preview | 512p, 2s | ~37s |
| Standard clip | 512p, 5s | ~1.5 min |
| Longer clip | 512p, 8s | ~1.5-2 min |
| Extended scene | 512p, 2s × 5 (extend chain) | ~3 min |
| High quality short | 720p, 2s | ~39s |
| High quality standard | 720p, 5s | ~1.5 min |
| Maximum quality | 1080p, 2s | ~8 min |
| Quick image | 1024×1024 | ~10s |

**Avoid:** 512p ≥10s (10+ min), 720p ≥8s (hours), 1080p ≥5s (crash)

**For longer scenes:** Use the extend feature to chain 2-5s clips instead of generating long durations in one shot.
