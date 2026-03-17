"""
tools.wan_animator
~~~~~~~~~~~~~~~~~~
Generate short video clips from still images using Stable Video Diffusion (SVD).

SVD is Stability AI's image-to-video model. The SVD-XT variant generates 25
frames (~1s at 25fps) of genuine motion from a single conditioning image 
camera movements, particle effects, atmospheric shifts, and object animation.
At ~4.2GB in fp16, it fits comfortably on Apple M4 Pro with 24GB unified memory.

For each scene, we generate an SVD clip from the primary image, then
composite the result with color grading, film grain, and fade effects
via FFmpeg to match the target narration duration.

Falls back to the multi-image FFmpeg animator if SVD fails.

Inputs
------
- scenes     : list[dict] – Scenes with ``image_path(s)`` and ``narration_duration``.
- output_dir : str | Path – Where to save clips.
- fps        : int        – Output frame rate.

Outputs
-------
list[dict] – Scenes with ``clip_path`` added.
"""

from __future__ import annotations

import gc
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_svd_pipe = None


def _load_svd_pipeline():
    """Load Stable Video Diffusion XT pipeline (~4.2 GB fp16)."""
    global _svd_pipe
    if _svd_pipe is not None:
        return _svd_pipe

    import torch
    from diffusers import StableVideoDiffusionPipeline

    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"

    logger.info("Loading SVD-XT image-to-video pipeline …")

    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # SVD works in fp16 on MPS (unlike SDXL which needs float32)
        pipe = pipe.to(device)

        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        _svd_pipe = pipe
        logger.info("SVD-XT pipeline ready on %s.", device)
        return pipe

    except Exception as exc:
        logger.warning("Failed to load SVD-XT pipeline: %s", exc)
        raise


def cleanup_svd_pipeline():
    """Free SVD pipeline GPU memory."""
    global _svd_pipe
    if _svd_pipe is not None:
        del _svd_pipe
        _svd_pipe = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        logger.info("SVD-XT pipeline unloaded.")


def animate_scenes(
    scenes: list[dict[str, Any]],
    output_dir: str | Path = "output/clips",
    fps: int = 24,
    width: int = 1920,
    height: int = 1080,
    **_extra: Any,
) -> list[dict[str, Any]]:
    """Create video clips using Wan 2.1 image-to-video + FFmpeg compositing.

    Strategy:
    1. Generate a short Wan clip (2-4s of real motion) from the primary image
    2. Extend to full narration duration via speed ramp + loop + multi-image compositing
    3. Apply cinematic post-processing (color grade, grain, vignette, fades)

    Falls back to FFmpeg-only animation if Wan is unavailable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try loading SVD
    svd_available = False
    try:
        _load_svd_pipeline()
        svd_available = True
    except Exception as exc:
        logger.warning("SVD not available (%s). Using FFmpeg-only animation.", exc)

    results: list[dict[str, Any]] = []

    for scene in scenes:
        num = scene["scene_number"]
        clip_path = output_dir / f"clip_{num:03d}.mp4"
        duration = scene.get("narration_duration", 10.0)
        duration = max(duration + 0.5, 3.0)
        mood = scene.get("mood", "contemplative")

        image_paths = scene.get("image_paths", [])
        if not image_paths:
            single = scene.get("image_path")
            image_paths = [single] if single else []

        if not image_paths:
            logger.warning("Scene %d has no images, skipping.", num)
            continue

        n_imgs = len(image_paths)
        logger.info(
            "Animating scene %d (%s, %.1fs, %d images, svd=%s) → %s",
            num, mood, duration, n_imgs, svd_available, clip_path,
        )

        try:
            if svd_available:
                _create_svd_clip(
                    image_paths, clip_path, duration, fps, width, height,
                    mood, num,
                )
            else:
                _create_ffmpeg_clip(
                    image_paths, clip_path, duration, fps, width, height,
                    mood, num,
                )
        except Exception as exc:
            logger.warning("Animation failed for scene %d (%s), trying FFmpeg fallback.", num, exc)
            try:
                _create_ffmpeg_clip(
                    image_paths, clip_path, duration, fps, width, height,
                    mood, num,
                )
            except Exception as exc2:
                logger.error("FFmpeg fallback also failed for scene %d: %s", num, exc2)
                continue

        results.append({**scene, "clip_path": clip_path})

    if svd_available:
        cleanup_svd_pipeline()

    logger.info("Animation complete – %d clips.", len(results))
    return results


# ──────────────────────────────────────────────────────────────────────
# SVD clip generation
# ──────────────────────────────────────────────────────────────────────

def _create_svd_clip(
    image_paths: list,
    out: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
    mood: str,
    scene_num: int,
) -> None:
    """Generate a clip using SVD-XT for the primary image,
    then extend with multi-image compositing to fill the duration."""
    from PIL import Image
    import torch

    pipe = _load_svd_pipeline()

    # Load and resize primary image  SVD-XT works best at 1024x576
    # but we use 512x288 to fit in MPS memory with fp16
    primary_img = Image.open(str(image_paths[0])).convert("RGB")
    svd_w, svd_h = 512, 288
    primary_img = primary_img.resize((svd_w, svd_h), Image.LANCZOS)

    # Motion bucket controls amount of motion (1=minimal, 255=maximum)
    mood_motion = {
        "epic": 180, "dramatic": 200, "energetic": 220,
        "contemplative": 100, "warm": 120, "mysterious": 140, "serene": 80,
    }
    motion_bucket = mood_motion.get(mood, 127)

    logger.info("  Generating SVD clip for scene %d (motion=%d) …", scene_num, motion_bucket)
    with torch.no_grad():
        output = pipe(
            image=primary_img,
            width=svd_w,
            height=svd_h,
            num_frames=14,           # Reduced from 25 to fit in MPS memory
            decode_chunk_size=4,     # Smaller chunks to save memory
            motion_bucket_id=motion_bucket,
            noise_aug_strength=0.02,
            num_inference_steps=20,
        )

    # Export frames to a temp video
    svd_clip_path = out.with_suffix(".svd_raw.mp4")
    _export_frames(output.frames[0], svd_clip_path, fps)

    # Composite: loop SVD clip + additional images to fill duration
    _composite_final_clip(
        svd_clip_path, image_paths, out, duration, fps, width, height, mood, scene_num,
    )

    # Clean temp
    svd_clip_path.unlink(missing_ok=True)


def _export_frames(frames, output_path: Path, fps: int) -> None:
    """Export video output frames (list of PIL Images) to MP4."""
    import numpy as np

    # frames is a list of PIL images
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(frames):
            if hasattr(frame, 'save'):
                frame.save(f"{tmpdir}/frame_{i:04d}.png")
            else:
                from PIL import Image
                if isinstance(frame, np.ndarray):
                    Image.fromarray(frame).save(f"{tmpdir}/frame_{i:04d}.png")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", f"{tmpdir}/frame_%04d.png",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            check=True, capture_output=True,
        )


def _composite_final_clip(
    svd_clip: Path,
    image_paths: list,
    out: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
    mood: str,
    scene_num: int,
) -> None:
    """Build final clip: slow-motion SVD clip + multi-image transitions.

    Strategy:
    - Use the SVD clip (with genuine AI motion) as the hero segment
    - Slow it down 2-3x to stretch the motion
    - Crossfade to Ken Burns on additional images
    - Apply color grading, grain, vignette, and fades
    """
    color = _COLOR_GRADES.get(mood, _COLOR_GRADES["contemplative"])
    fade_out_start = max(duration - 0.8, 0.1)

    # Get SVD clip duration
    svd_dur = _get_duration(svd_clip)
    if svd_dur <= 0:
        svd_dur = 1.0

    # Calculate how much we need to slow the SVD clip to fill ~40% of duration
    svd_target = duration * 0.4
    slowdown = max(svd_dur / svd_target, 0.2)  # setpts multiplier
    speed_factor = 1.0 / slowdown

    n_imgs = len(image_paths)

    if n_imgs >= 2:
        # Strategy: SVD clip (slowed) → crossfade → Ken Burns on image 2 → crossfade → KB image 3
        remaining_dur = duration - svd_target
        trans_dur = min(1.0, remaining_dur / 4)
        img_segment_dur = (remaining_dur + trans_dur * min(n_imgs - 1, 2)) / max(n_imgs - 1, 1)
        img_frames = max(int(img_segment_dur * fps), fps)

        inputs = ["-i", str(svd_clip)]
        for ip in image_paths[1:3]:
            inputs.extend(["-loop", "1", "-i", str(ip)])

        parts = []
        # Slowed SVD clip
        parts.append(
            f"[0:v]setpts={1/speed_factor:.3f}*PTS,"
            f"scale={width}:{height}:flags=lanczos[svd]"
        )

        # Ken Burns on additional images
        moves = _CAMERA_MOVES.get(mood, _CAMERA_MOVES["contemplative"])
        for i in range(min(n_imgs - 1, 2)):
            mv = moves[(i + 1) % len(moves)].format(d=img_frames)
            parts.append(
                f"[{i+1}:v]scale=3072:1728:flags=lanczos,"
                f"zoompan={mv}:d={img_frames}:s={width}x{height}:fps={fps}[kb{i}]"
            )

        # Build xfade chain
        import random
        rng = random.Random(scene_num * 77)
        xfade_styles = ["dissolve", "smoothleft", "circlecrop", "fadeblack", "radial"]

        current = "[svd]"
        for i in range(min(n_imgs - 1, 2)):
            t = rng.choice(xfade_styles)
            offset = svd_target + i * (img_segment_dur - trans_dur) - trans_dur * i
            offset = max(offset, 0.5)
            out_label = f"[x{i}]" if i < min(n_imgs - 1, 2) - 1 else "[merged]"
            parts.append(
                f"{current}[kb{i}]xfade=transition={t}:duration={trans_dur:.3f}"
                f":offset={offset:.3f}{out_label}"
            )
            current = out_label

        if min(n_imgs - 1, 2) == 0:
            parts.append("[svd]copy[merged]")

        # Post-processing
        parts.append(
            f"[merged]noise=alls=4:allf=t,{color},vignette=PI/4,"
            f"fade=t=in:d=0.8,fade=t=out:st={fade_out_start:.2f}:d=0.8[out]"
        )

        fc = ";".join(parts)
        subprocess.run(
            [
                "ffmpeg", "-y",
                *inputs,
                "-filter_complex", fc,
                "-map", "[out]",
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", str(fps),
                str(out),
            ],
            check=True, capture_output=True,
        )
    else:
        # Single image: just slow SVD clip to fill duration
        vf = (
            f"setpts={1/speed_factor:.3f}*PTS,"
            f"scale={width}:{height}:flags=lanczos,"
            f"noise=alls=4:allf=t,{color},vignette=PI/4,"
            f"fade=t=in:d=0.8,fade=t=out:st={fade_out_start:.2f}:d=0.8"
        )
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-stream_loop", "-1",
                "-i", str(svd_clip),
                "-vf", vf,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", str(fps),
                str(out),
            ],
            check=True, capture_output=True,
        )


# ──────────────────────────────────────────────────────────────────────
# FFmpeg-only fallback (multi-image Ken Burns with transitions)
# ──────────────────────────────────────────────────────────────────────

_CAMERA_MOVES = {
    "epic": [
        "z='1.0+0.15*(on/{d})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        "z='1.1':x='(iw-iw/zoom)*(on/{d})':y='ih/2-(ih/zoom/2)'",
        "z='if(eq(on,1),1.2,max(1.2-0.15*(on/{d}),1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    ],
    "contemplative": [
        "z='1.0+0.08*(on/{d})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        "z='1.06':x='(iw-iw/zoom)/2+sin(on*0.003)*20':y='(ih-ih/zoom)/2'",
        "z='if(eq(on,1),1.12,max(1.12-0.08*(on/{d}),1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    ],
    "dramatic": [
        "z='1.0+0.20*(on/{d})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        "z='1.12':x='(iw-iw/zoom)*(1-on/{d})':y='ih/2-(ih/zoom/2)'",
        "z='1.05+0.10*(on/{d})':x='(iw-iw/zoom)*0.3*(on/{d})':y='(ih-ih/zoom)*0.3*(on/{d})'",
    ],
    "warm": [
        "z='if(eq(on,1),1.15,max(1.15-0.12*(on/{d}),1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        "z='1.04+0.06*(on/{d})':x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-0.3*(on/{d}))'",
        "z='1.08':x='(iw-iw/zoom)/2+sin(on*0.005)*15':y='(ih-ih/zoom)/2+cos(on*0.005)*10'",
    ],
    "mysterious": [
        "z='1.0+0.18*(on/{d})':x='iw/2-(iw/zoom/2)+sin(on*0.008)*15':y='ih/2-(ih/zoom/2)'",
        "z='1.1':x='(iw-iw/zoom)/2+sin(on*0.004)*25':y='(ih-ih/zoom)/2+cos(on*0.006)*12'",
        "z='if(eq(on,1),1.18,max(1.18-0.12*(on/{d}),1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    ],
    "energetic": [
        "z='1.0+0.22*(on/{d})':x='(iw-iw/zoom)*0.5*(on/{d})':y='(ih-ih/zoom)*0.3*(on/{d})'",
        "z='1.12':x='(iw-iw/zoom)*(on/{d})':y='ih/2-(ih/zoom/2)'",
        "z='1.05+0.12*(on/{d})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    ],
    "serene": [
        "z='1.0+0.06*(on/{d})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        "z='1.05':x='(iw-iw/zoom)/2+sin(on*0.002)*12':y='(ih-ih/zoom)/2+cos(on*0.002)*8'",
        "z='if(eq(on,1),1.10,max(1.10-0.06*(on/{d}),1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    ],
}

_COLOR_GRADES = {
    "epic":          "eq=brightness=0.02:saturation=1.3:contrast=1.05,colorbalance=rs=0.05:gs=-0.02:bs=0.08",
    "contemplative": "eq=brightness=-0.02:saturation=0.9:contrast=1.05,colorbalance=rs=-0.03:gs=0.02:bs=0.06",
    "dramatic":      "eq=brightness=-0.04:contrast=1.15:saturation=1.2,colorbalance=rs=0.06:gs=-0.03:bs=-0.02",
    "warm":          "eq=brightness=0.03:saturation=1.1,colorbalance=rs=0.08:gs=0.04:bs=-0.06",
    "mysterious":    "eq=brightness=-0.06:saturation=0.85:contrast=1.1,colorbalance=rs=-0.02:gs=-0.02:bs=0.08",
    "energetic":     "eq=brightness=0.02:saturation=1.4:contrast=1.05,colorbalance=rs=0.04:gs=0.02:bs=-0.02",
    "serene":        "eq=brightness=0.01:saturation=0.95,colorbalance=rs=-0.02:gs=0.04:bs=0.06",
}

_XFADE_STYLES = [
    "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circlecrop", "dissolve", "fadeblack", "radial",
    "horzopen", "vertopen", "wipeleft", "wiperight",
]


def _create_ffmpeg_clip(
    image_paths: list,
    out: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
    mood: str,
    scene_num: int,
) -> None:
    """Multi-image FFmpeg animation fallback (3 images with transitions)."""
    import random

    n_imgs = min(len(image_paths), 3)
    color = _COLOR_GRADES.get(mood, _COLOR_GRADES["contemplative"])
    fade_out_start = max(duration - 0.8, 0.1)

    if n_imgs >= 2:
        trans_dur = min(1.2, duration / 6)
        segment_dur = (duration + (n_imgs - 1) * trans_dur) / n_imgs
        frames_per_seg = max(int(segment_dur * fps), fps)
        seg_time = frames_per_seg / fps

        moves = _CAMERA_MOVES.get(mood, _CAMERA_MOVES["contemplative"])
        rng = random.Random(scene_num * 42)
        upscale = "scale=3072:1728:flags=lanczos"

        zp_parts = []
        for i in range(n_imgs):
            mv = moves[i % len(moves)].format(d=frames_per_seg)
            zp_parts.append(
                f"[{i}:v]{upscale},"
                f"zoompan={mv}:d={frames_per_seg}:s={width}x{height}:fps={fps}[z{i}]"
            )

        xfade_parts = []
        current = "[z0]"
        for i in range(1, n_imgs):
            t = rng.choice(_XFADE_STYLES)
            offset = i * seg_time - i * trans_dur
            offset = max(offset, 0.1)
            out_label = f"[x{i}]" if i < n_imgs - 1 else "[merged]"
            xfade_parts.append(
                f"{current}[z{i}]xfade=transition={t}:duration={trans_dur:.3f}"
                f":offset={offset:.3f}{out_label}"
            )
            current = out_label

        post = (
            f"[merged]noise=alls=4:allf=t,{color},vignette=PI/4,"
            f"fade=t=in:d=0.8,fade=t=out:st={fade_out_start:.2f}:d=0.8[out]"
        )

        fc = ";".join(zp_parts + xfade_parts + [post])

        inputs = []
        for ip in image_paths[:n_imgs]:
            inputs.extend(["-loop", "1", "-i", str(ip)])

        subprocess.run(
            [
                "ffmpeg", "-y",
                *inputs,
                "-filter_complex", fc,
                "-map", "[out]",
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", str(fps),
                str(out),
            ],
            check=True, capture_output=True,
        )
    else:
        # Single image fallback
        total_frames = max(int(duration * fps), fps)
        moves = _CAMERA_MOVES.get(mood, _CAMERA_MOVES["contemplative"])
        mv = moves[scene_num % len(moves)].format(d=total_frames)

        vf = (
            f"scale=3072:1728:flags=lanczos,"
            f"zoompan={mv}:d={total_frames}:s={width}x{height}:fps={fps},"
            f"noise=alls=4:allf=t,{color},vignette=PI/4,"
            f"fade=t=in:d=0.8,fade=t=out:st={fade_out_start:.2f}:d=0.8"
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(image_paths[0]),
                "-vf", vf,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", str(fps),
                str(out),
            ],
            check=True, capture_output=True,
        )


def _get_duration(path: Path) -> float:
    """Get video duration via ffprobe."""
    try:
        import json
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        return 0.0
