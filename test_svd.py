#!/usr/bin/env python3
"""Quick SVD test: fp16 UNet + float32 VAE on MPS."""

import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np

# Load in fp16
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("mps")
pipe.enable_attention_slicing()

# Upcast VAE to float32 to avoid NaN in decoding
pipe.vae = pipe.vae.to(dtype=torch.float32)

img = Image.open("output/render_images/scene_001_v1.png").convert("RGB")
img = img.resize((512, 288), Image.LANCZOS)

print("Running SVD with fp16 UNet + float32 VAE...")
with torch.no_grad():
    output = pipe(
        image=img,
        width=512,
        height=288,
        num_frames=14,
        decode_chunk_size=4,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        num_inference_steps=20,
    )

frames = output.frames[0]
arr = np.array(frames[0])
print(f"Frame 0: shape={arr.shape}, min={arr.min()}, max={arr.max()}, "
      f"mean={arr.mean():.1f}, std={arr.std():.1f}")
black_pct = (arr == 0).all(axis=2).mean() * 100
print(f"Black pixels: {black_pct:.1f}%")
frames[0].save("/tmp/svd_test2.png")
print(f"Got {len(frames)} frames. Saved first to /tmp/svd_test2.png")
