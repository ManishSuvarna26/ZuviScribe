"""
tools.documentary_animator
~~~~~~~~~~~~~~~~~~~~~~~~~~
Create cinematic animated clips from SDXL images for documentary-style videos.

Uses a layered compositing approach via FFmpeg to create the illusion of
motion and depth from static images:

1. **Dynamic Ken Burns** – Camera zooms, pans, and drift with easing curves
2. **Parallax layers** – Foreground/background separation via crop offsets
3. **Vignette & color grading** – Mood-specific color treatments
4. **Particle overlays** – Subtle dust/light leak effects
5. **Smooth transitions** – Each clip has fade-in/fade-out built in

Inputs
------
- scenes     : list[dict] – Scenes with ``image_path`` and ``narration_duration``.
- output_dir : str | Path – Where to save clips.
- fps        : int        – Output frame rate.

Outputs
-------
list[dict] – Scenes with ``clip_path`` added.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Camera movement presets  cinematic Ken Burns with easing
# ──────────────────────────────────────────────────────────────────────
# Each preset is a zoompan filter with different motion characteristics.
# {d} = total frames, {s} = output size, {fps} = frame rate

_CAMERA_PRESETS = {
    "epic": [
        # Dramatic slow zoom in to center
        "zoompan=z='1.0+0.15*on/{d}':d={d}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
        # Sweeping pan left-to-right with slight zoom
        "zoompan=z='1.05+0.05*on/{d}':d={d}:x='(iw-iw/zoom)*on/{d}':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
    ],
    "contemplative": [
        # Very slow drift down with gentle zoom
        "zoompan=z='1.0+0.08*on/{d}':d={d}:x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*0.2*on/{d}':s={s}:fps={fps}",
        # Ultra-slow zoom out (reveals more)
        "zoompan=z='if(eq(on,1),1.15,max(1.15-0.12*on/{d},1.0))':d={d}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
    ],
    "dramatic": [
        # Fast zoom in (punch effect)
        "zoompan=z='1.0+0.20*on/{d}':d={d}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
        # Pan right-to-left with zoom
        "zoompan=z='1.08+0.07*on/{d}':d={d}:x='(iw-iw/zoom)*(1-on/{d})':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
    ],
    "warm": [
        # Gentle zoom out from center
        "zoompan=z='if(eq(on,1),1.12,max(1.12-0.10*on/{d},1.0))':d={d}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
        # Slow drift up-right
        "zoompan=z='1.06':d={d}:x='(iw-iw/zoom)*0.3+on*0.15':y='(ih-ih/zoom)*(1-0.3*on/{d})':s={s}:fps={fps}",
    ],
    "mysterious": [
        # Slow creeping zoom
        "zoompan=z='1.0+0.18*on/{d}':d={d}:x='iw/2-(iw/zoom/2)+sin(on*0.01)*20':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
        # Drift with slight oscillation
        "zoompan=z='1.08':d={d}:x='(iw-iw/zoom)/2+sin(on*0.005)*30':y='(ih-ih/zoom)/2+cos(on*0.005)*15':s={s}:fps={fps}",
    ],
    "energetic": [
        # Fast pan
        "zoompan=z='1.10':d={d}:x='(iw-iw/zoom)*on/{d}':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
        # Zoom in + pan
        "zoompan=z='1.0+0.22*on/{d}':d={d}:x='(iw-iw/zoom)*0.5*on/{d}':y='(ih-ih/zoom)*0.3*on/{d}':s={s}:fps={fps}",
    ],
    "serene": [
        # Ultra-slow zoom in
        "zoompan=z='1.0+0.06*on/{d}':d={d}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={s}:fps={fps}",
        # Gentle float
        "zoompan=z='1.04':d={d}:x='(iw-iw/zoom)/2+sin(on*0.003)*10':y='(ih-ih/zoom)/2+cos(on*0.003)*8':s={s}:fps={fps}",
    ],
}

# Mood-specific color grading (FFmpeg eq/colorbalance filters)
_COLOR_GRADES = {
    "epic": "eq=brightness=0.02:saturation=1.3,colorbalance=rs=0.05:gs=-0.02:bs=0.08",
    "contemplative": "eq=brightness=-0.02:saturation=0.9,colorbalance=rs=-0.03:gs=0.02:bs=0.06",
    "dramatic": "eq=brightness=-0.04:contrast=1.15:saturation=1.2,colorbalance=rs=0.06:gs=-0.03:bs=-0.02",
    "warm": "eq=brightness=0.03:saturation=1.1,colorbalance=rs=0.08:gs=0.04:bs=-0.06",
    "mysterious": "eq=brightness=-0.06:saturation=0.85:contrast=1.1,colorbalance=rs=-0.02:gs=-0.02:bs=0.08",
    "energetic": "eq=brightness=0.02:saturation=1.4:contrast=1.05,colorbalance=rs=0.04:gs=0.02:bs=-0.02",
    "serene": "eq=brightness=0.01:saturation=0.95,colorbalance=rs=-0.02:gs=0.04:bs=0.06",
}


def animate_scenes(
    scenes: list[dict[str, Any]],
    output_dir: str | Path = "output/clips",
    fps: int = 24,
    width: int = 1920,
    height: int = 1080,
    **_extra: Any,
) -> list[dict[str, Any]]:
    """Create cinematic animated clips for each scene.

    Duration is driven by ``narration_duration`` on each scene dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for scene in scenes:
        num = scene["scene_number"]
        img = Path(scene["image_path"])
        clip_path = output_dir / f"clip_{num:03d}.mp4"
        duration = scene.get("narration_duration", scene.get("duration_hint", 10.0))
        # Add 0.5s padding for breathing room
        duration = max(duration + 0.5, 2.0)
        mood = scene.get("mood", "contemplative")

        logger.info("Animating scene %d (%s, %.1fs) → %s", num, mood, duration, clip_path)
        _create_cinematic_clip(img, clip_path, duration, fps, width, height, mood, num)
        results.append({**scene, "clip_path": clip_path})

    logger.info("Animation complete – %d clips.", len(results))
    return results


def _create_cinematic_clip(
    image: Path,
    out: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
    mood: str,
    scene_num: int,
) -> None:
    """Create a single cinematic clip with camera movement + color grading + vignette."""
    total_frames = int(duration * fps)
    if total_frames < fps:
        total_frames = fps

    size = f"{width}x{height}"

    # Pick camera movement based on mood + scene number for variety
    presets = _CAMERA_PRESETS.get(mood, _CAMERA_PRESETS["contemplative"])
    move_template = presets[scene_num % len(presets)]
    zoompan = move_template.format(d=total_frames, s=size, fps=fps)

    # Color grading
    color = _COLOR_GRADES.get(mood, _COLOR_GRADES["contemplative"])

    # Vignette effect (subtle darkening at edges)
    vignette = "vignette=PI/4:mode=forward"

    # Fade in/out (0.8s each)
    fade_frames = int(0.8 * fps)
    fade_in = f"fade=t=in:st=0:d=0.8"
    fade_out = f"fade=t=out:st={duration - 0.8}:d=0.8"

    # Build complete filter chain
    vf = f"{zoompan},{color},{vignette},{fade_in},{fade_out}"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(image),
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            str(out),
        ],
        check=True,
        capture_output=True,
    )
