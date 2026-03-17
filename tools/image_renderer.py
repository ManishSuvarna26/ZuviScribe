"""
tools.image_renderer
~~~~~~~~~~~~~~~~~~~~
Render high-resolution images for each storyboard scene using
Stable Diffusion XL via the HuggingFace *diffusers* library on Apple MPS.

Falls back to a PIL-generated art card only if SDXL completely fails to load.

Inputs
------
- scenes      : list[dict] – Storyboard scenes from ``storyboard_generator``.
- output_dir  : str | Path – Directory to save rendered images.
- model       : str        – HuggingFace model ID (default SDXL-Turbo for speed).
- width       : int        – Image width in pixels (default 1024).
- height      : int        – Image height in pixels (default 576).
- steps       : int        – Diffusion inference steps.
- cfg_scale   : float      – Classifier-free guidance scale.
- negative_prompt : str    – Negative prompt applied to all generations.

Outputs
-------
list[dict] – Scenes with ``image_path`` added.
"""

from __future__ import annotations

import hashlib
import logging
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level pipeline cache so we only load the model once.
_sdxl_pipe = None


def cleanup_pipeline():
    """Explicitly unload SDXL pipeline to free GPU memory."""
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        del _sdxl_pipe
        _sdxl_pipe = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        logger.info("SDXL pipeline unloaded.")


def _load_sdxl_pipeline(model_id: str):
    """Load SDXL pipeline once, caching it for subsequent scenes."""
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        return _sdxl_pipe

    import torch
    from diffusers import AutoPipelineForText2Image

    logger.info("Loading SDXL pipeline: %s (this downloads ~6 GB on first run) …", model_id)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load fp16 weights (smaller download), then upcast to float32 for MPS
    # (float16 on MPS produces NaN in UNet/VAE)
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to(dtype=torch.float32, device=device)

    # Memory optimisations
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    _sdxl_pipe = pipe
    logger.info("SDXL pipeline ready on %s.", device)
    return pipe


def render_images(
    scenes: list[dict[str, Any]],
    output_dir: str | Path = "output/images",
    model: str = "stabilityai/sdxl-turbo",
    width: int = 1024,
    height: int = 576,
    steps: int = 4,
    cfg_scale: float = 0.0,
    negative_prompt: str = "blurry, low quality, watermark, text, words, letters, ugly, deformed",
    variants_per_scene: int = 3,
    **_extra: Any,
) -> list[dict[str, Any]]:
    """Generate multiple image variants per scene for richer animation.

    Returns scenes with ``image_path`` (first variant) and ``image_paths``
    (all variants) added.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = None
    try:
        pipe = _load_sdxl_pipeline(model)
    except Exception as exc:
        logger.error("Could not load SDXL pipeline (%s). Falling back to PIL cards.", exc)

    results: list[dict[str, Any]] = []

    for scene in scenes:
        num = scene["scene_number"]
        raw_prompt = scene.get("visual_prompt", scene.get("description", ""))

        # Generate variant prompts (wide, detail, abstract)
        variant_prompts = _make_variant_prompts(raw_prompt, variants_per_scene)
        image_paths: list[Path] = []

        for vi, vprompt in enumerate(variant_prompts):
            img_path = output_dir / f"scene_{num:03d}_v{vi + 1}.png"
            prompt = _enhance_prompt(vprompt)

            logger.info("Rendering scene %d variant %d/%d …", num, vi + 1, len(variant_prompts))

            if pipe is not None:
                try:
                    import torch
                    import numpy as np
                    with torch.no_grad():
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt if cfg_scale > 0 else None,
                            num_inference_steps=steps,
                            guidance_scale=cfg_scale,
                            width=width,
                            height=height,
                        )
                    image = result.images[0]
                    arr = np.array(image)
                    if arr.std() < 1.0:
                        logger.warning("Blank image scene %d v%d (std=%.2f). PIL fallback.", num, vi + 1, arr.std())
                        _generate_pil_card(raw_prompt, num, img_path, width, height)
                    else:
                        image.save(str(img_path))
                        logger.info("SDXL scene %d v%d → %s (std=%.1f)", num, vi + 1, img_path.name, arr.std())
                except Exception as exc:
                    logger.warning("SDXL failed scene %d v%d (%s), PIL fallback.", num, vi + 1, exc)
                    _generate_pil_card(raw_prompt, num, img_path, width, height)
            else:
                _generate_pil_card(raw_prompt, num, img_path, width, height)

            image_paths.append(img_path)

        results.append({
            **scene,
            "image_path": image_paths[0],       # backwards compat
            "image_paths": image_paths,          # all variants
        })

    return results


def _make_variant_prompts(base: str, n: int) -> list[str]:
    """Generate N visual prompt variants from a base prompt.

    Variant 1: Wide establishing shot (base as-is)
    Variant 2: Close-up detail focus
    Variant 3: Abstract atmospheric interpretation
    """
    if n <= 1:
        return [base]
    variants = [base]
    if n >= 2:
        variants.append(
            f"Extreme close-up detail shot, {base}, "
            f"shallow depth of field, macro photography, intricate details"
        )
    if n >= 3:
        variants.append(
            f"Abstract atmospheric interpretation, {base}, "
            f"bokeh light particles, ethereal glow, dreamy soft focus"
        )
    return variants[:n]


def _enhance_prompt(prompt: str) -> str:
    """Add quality keywords while keeping under CLIP's 77-token limit."""
    words = prompt.split()
    if len(words) > 50:
        prompt = " ".join(words[:50])
    return f"{prompt}, cinematic lighting, 8k, photorealistic, volumetric light"


# ──────────────────────────────────────────────────────────────────────
# PIL art-card fallback (used only when SDXL is completely unavailable)
# ──────────────────────────────────────────────────────────────────────

# Palette of vibrant gradient colour pairs keyed by hash of scene number
_PALETTES = [
    ((15, 32, 80), (90, 10, 180)),    # deep blue → violet
    ((180, 15, 80), (255, 100, 20)),  # crimson → orange
    ((10, 120, 90), (20, 200, 160)),  # forest → teal
    ((70, 10, 130), (200, 50, 200)),  # indigo → magenta
    ((10, 60, 130), (20, 180, 220)),  # navy → cyan
    ((130, 60, 10), (220, 170, 20)),  # brown → gold
    ((10, 100, 40), (80, 200, 80)),   # dark green → lime
    ((80, 10, 50), (200, 60, 130)),   # plum → pink
]


def _generate_pil_card(
    description: str,
    scene_number: int,
    output_path: Path,
    width: int = 1024,
    height: int = 576,
) -> None:
    """Create a stylised animated art card using PIL.

    Generates a left-to-right gradient background with the scene number badge
    and a word-wrapped description overlay.  Visually interesting and gives
    the Ken Burns animator something colourful to work with.
    """
    from PIL import Image, ImageDraw, ImageFont

    palette_idx = int(hashlib.md5(str(scene_number).encode()).hexdigest(), 16) % len(_PALETTES)
    color_left, color_right = _PALETTES[palette_idx]

    # Build gradient background
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for x in range(width):
        t = x / (width - 1)
        r = int(color_left[0] + t * (color_right[0] - color_left[0]))
        g = int(color_left[1] + t * (color_right[1] - color_left[1]))
        b = int(color_left[2] + t * (color_right[2] - color_left[2]))
        for y in range(height):
            pixels[x, y] = (r, g, b)

    draw = ImageDraw.Draw(img)

    # Semi-transparent overlay strip for text legibility
    overlay_top = height // 3
    for y in range(overlay_top, height):
        alpha = min(1.0, (y - overlay_top) / (height - overlay_top) * 1.6)
        for x in range(width):
            orig = pixels[x, y]
            blended = tuple(int(c * (1 - alpha * 0.65)) for c in orig)
            pixels[x, y] = blended  # type: ignore[assignment]

    # Scene badge
    badge_text = f"SCENE {scene_number}"
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        font_body  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except OSError:
        font_large = ImageFont.load_default()
        font_body  = font_large

    draw.text((48, 48), badge_text, fill=(255, 255, 255), font=font_large)

    # Word-wrapped description
    wrapped = textwrap.fill(description, width=60)
    draw.multiline_text(
        (48, overlay_top + 32),
        wrapped,
        fill=(230, 230, 230),
        font=font_body,
        spacing=8,
    )

    img.save(str(output_path), "PNG")
