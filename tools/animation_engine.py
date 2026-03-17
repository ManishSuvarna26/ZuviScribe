"""
tools.animation_engine
~~~~~~~~~~~~~~~~~~~~~~
Turn static scene images into short animated video clips.

Supports two strategies:
* **animatediff** – Uses the HuggingFace diffusers AnimateDiff pipeline
  with a SD 1.5 base model + motion adapter.  Produces genuine AI-animated
  clips.  Falls back to an enhanced Ken Burns effect if it fails.
* **img2vid** – FFmpeg Ken Burns with varied camera movements (zoom, pan,
  parallax).  Lightweight, no GPU required.

Inputs
------
- scenes           : list[dict] – Scenes with ``image_path`` from ``image_renderer``.
- output_dir       : str | Path – Directory to save animated clips.
- method           : str        – ``"animatediff"`` or ``"img2vid"``.
- frames_per_scene : int        – Number of video frames per clip.
- fps              : int        – Frames per second for output clips.
- motion_strength  : float      – Motion intensity (0.0 – 1.0).

Outputs
-------
list[dict] – Scenes augmented with ``clip_path``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level cache for the AnimateDiff pipeline
_animatediff_pipe = None


def _load_animatediff_pipeline():
    """Load AnimateDiff pipeline once, cache globally."""
    global _animatediff_pipe
    if _animatediff_pipe is not None:
        return _animatediff_pipe

    import torch
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from diffusers.utils import export_to_video

    logger.info("Loading AnimateDiff pipeline (downloads models on first run) …")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    # Use AnimateDiff-Lightning for faster generation
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=dtype,
    )

    pipe = AnimateDiffPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=dtype,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
    )
    pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    _animatediff_pipe = pipe
    logger.info("AnimateDiff pipeline ready on %s.", device)
    return pipe


def animate_scenes(
    scenes: list[dict[str, Any]],
    output_dir: str | Path = "output/clips",
    method: str = "animatediff",
    frames_per_scene: int = 16,
    fps: int = 8,
    motion_strength: float = 0.6,
    **_extra: Any,
) -> list[dict[str, Any]]:
    """Generate a short animated clip for each scene image.

    With ``method="animatediff"``, uses the HuggingFace diffusers AnimateDiff
    pipeline for genuine AI-generated motion.  Falls back to an enhanced
    multi-movement Ken Burns effect via FFmpeg.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try loading AnimateDiff once, use Ken Burns if it fails
    use_animatediff = False
    if method == "animatediff":
        try:
            _load_animatediff_pipeline()
            use_animatediff = True
        except Exception as exc:
            logger.warning("AnimateDiff load failed (%s). Using enhanced Ken Burns.", exc)

    results: list[dict[str, Any]] = []

    for scene in scenes:
        num = scene["scene_number"]
        img = Path(scene["image_path"])
        clip_path = output_dir / f"clip_{num:03d}.mp4"
        duration = scene.get("duration_hint", frames_per_scene / fps)

        logger.info("Animating scene %d → %s", num, clip_path)

        if use_animatediff:
            try:
                _generate_animatediff(
                    scene["description"], clip_path, frames_per_scene, fps
                )
            except Exception as exc:
                logger.warning("AnimateDiff failed for scene %d (%s), using Ken Burns.", num, exc)
                _ken_burns_enhanced(img, clip_path, duration, fps, num)
        else:
            _ken_burns_enhanced(img, clip_path, duration, fps, num)

        results.append({**scene, "clip_path": clip_path})

    logger.info("Animation complete – %d clips.", len(results))
    return results


# ──────────────────────────────────────────────────────────────────────
# AnimateDiff
# ──────────────────────────────────────────────────────────────────────

def _generate_animatediff(
    prompt: str,
    out: Path,
    n_frames: int,
    fps: int,
) -> None:
    """Generate an animated clip using the AnimateDiff pipeline."""
    import torch
    from diffusers.utils import export_to_video

    pipe = _animatediff_pipe

    enhanced_prompt = (
        f"{prompt}, masterpiece, best quality, highly detailed, "
        f"vibrant colors, smooth animation, fluid motion"
    )

    with torch.no_grad():
        result = pipe(
            prompt=enhanced_prompt,
            negative_prompt="blurry, low quality, static, frozen, ugly, deformed, text, watermark",
            num_frames=n_frames,
            guidance_scale=7.5,
            num_inference_steps=20,
            width=512,
            height=288,
        )

    export_to_video(result.frames[0], str(out), fps=fps)
    logger.info("AnimateDiff rendered → %s", out)


# ──────────────────────────────────────────────────────────────────────
# Enhanced Ken Burns (FFmpeg)  varied camera movements per scene
# ──────────────────────────────────────────────────────────────────────

# Different camera movement patterns for visual variety
_CAMERA_MOVES = [
    # Slow zoom in (classic Ken Burns)
    "zoompan=z='min(zoom+0.001,1.12)':d={d}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
    # Slow zoom out
    "zoompan=z='if(eq(on,1),1.12,max(zoom-0.001,1.0))':d={d}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
    # Pan left to right
    "zoompan=z='1.08':d={d}:x='(iw-iw/zoom)*on/{d}':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
    # Pan right to left
    "zoompan=z='1.08':d={d}:x='(iw-iw/zoom)*(1-on/{d})':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
    # Zoom in + drift up-left
    "zoompan=z='min(zoom+0.0012,1.15)':d={d}:x='(iw/2-(iw/zoom/2))-on*0.3':y='(ih/2-(ih/zoom/2))-on*0.15':s={s}:fps={fps}",
    # Zoom in + drift down-right
    "zoompan=z='min(zoom+0.0012,1.15)':d={d}:x='(iw/2-(iw/zoom/2))+on*0.25':y='(ih/2-(ih/zoom/2))+on*0.12':s={s}:fps={fps}",
]


def _ken_burns_enhanced(image: Path, out: Path, duration: float, fps: int, scene_num: int) -> None:
    """Create a varied Ken Burns clip  different scenes get different camera moves."""
    total_frames = int(duration * fps)
    if total_frames < 1:
        total_frames = fps  # minimum 1 second

    move = _CAMERA_MOVES[scene_num % len(_CAMERA_MOVES)]
    vf = move.format(d=total_frames, s="1024x576", fps=fps)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(image),
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
