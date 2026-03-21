"""Test FLUX.2 Klein 9B with bitsandbytes NF4 quantization + LoRA.

Run: cd backend && uv run python scripts/test_flux_klein_nf4.py
"""

import time
import gc
import torch
from pathlib import Path

MODEL_PATH = "C:/Users/taskm/AppData/Local/LTXDesktop/models/FLUX.2-klein-base-9B"
LORA_PATH = "C:/Users/taskm/AppData/Local/LTXDesktop/models/loras/jRB4slNlO3KYd18ROU5Up_pytorch_lora_weights_comfy_converted.safetensors"
OUTPUT_PATH = "D:/git/directors-desktop/backend/outputs/test_nf4_lora.png"

def main():
    from diffusers import BitsAndBytesConfig, Flux2KleinPipeline, AutoencoderKL, PipelineQuantizationConfig
    import numpy as np
    from PIL import Image

    print("=" * 60)
    print("FLUX.2 Klein 9B — NF4 Quantization + LoRA Test")
    print("=" * 60)

    # Step 1: Load pipeline with NF4 quantization
    print("\n[1/5] Loading pipeline with NF4 quantization...")
    t0 = time.time()

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    pipeline_quant_config = PipelineQuantizationConfig(
        quant_mapping={"transformer": nf4_config},
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_PATH,
        quantization_config=pipeline_quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # NF4 transformer (~5GB) fits on GPU, but T5-XXL text_encoder (~9GB bf16)
    # pushes total to ~21GB leaving no room for activations at 1024x1024.
    # CPU offload moves text_encoder off GPU after encoding, so the NF4
    # transformer gets full VRAM for inference.
    pipe.enable_model_cpu_offload()

    load_time = time.time() - t0
    print(f"    Pipeline loaded in {load_time:.1f}s")

    # Check VRAM usage after loading
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"    VRAM after load: {vram_mb:.0f} MB")

    # Step 2: Load LoRA
    print("\n[2/5] Loading LoRA...")
    t1 = time.time()
    pipe.load_lora_weights(LORA_PATH, adapter_name="user_lora")
    pipe.set_adapters(["user_lora"], adapter_weights=[1.0])
    lora_time = time.time() - t1
    print(f"    LoRA loaded in {lora_time:.1f}s")

    vram_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"    VRAM after LoRA: {vram_mb:.0f} MB")

    # Step 3: Generate (latent output to avoid VAE segfault)
    print("\n[3/5] Generating image (1024x1024, 28 steps)...")
    prompt = "DC animation style,with bold outlines,cel-shaded & muted color palette, A powerful superhero standing on a city rooftop at sunset, dramatic lighting, cape flowing in the wind, Gotham-style cityscape in background"

    generator = torch.Generator(device="cpu").manual_seed(42)

    t2 = time.time()
    output = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.0,
        num_inference_steps=28,
        generator=generator,
        output_type="latent",
        return_dict=True,
    )
    gen_time = time.time() - t2
    print(f"    Inference completed in {gen_time:.1f}s")

    latents = output.images.to("cpu")

    # Step 4: Destroy pipeline, decode with fresh VAE
    print("\n[4/5] Decoding latents with fresh VAE on CPU...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    t3 = time.time()
    vae_path = str(Path(MODEL_PATH) / "vae")
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32)
    vae = vae.to("cpu")
    vae.eval()

    latents_f32 = latents.to(dtype=torch.float32)
    with torch.no_grad():
        decoded = vae.decode(latents_f32, return_dict=False)[0]

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    arr = decoded[0].permute(1, 2, 0).numpy()
    pil_img = Image.fromarray((arr * 255).astype(np.uint8))
    decode_time = time.time() - t3
    print(f"    VAE decode completed in {decode_time:.1f}s")

    # Step 5: Save
    pil_img.save(OUTPUT_PATH)
    print(f"\n[5/5] Saved to {OUTPUT_PATH}")

    del vae, latents, latents_f32, decoded
    gc.collect()

    total = load_time + lora_time + gen_time + decode_time
    print("\n" + "=" * 60)
    print(f"RESULTS:")
    print(f"  Pipeline load (NF4): {load_time:.1f}s")
    print(f"  LoRA load:           {lora_time:.1f}s")
    print(f"  Inference (28 steps): {gen_time:.1f}s")
    print(f"  VAE decode (CPU):    {decode_time:.1f}s")
    print(f"  TOTAL:               {total:.1f}s")
    print(f"  Peak VRAM:           {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
