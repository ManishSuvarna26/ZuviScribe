"""
tools.video_assembler
~~~~~~~~~~~~~~~~~~~~~
Assemble animated clips and narration audio into the final output video.

Supports two back-ends:

* **FFmpeg-only** (default) – concatenates clips with cross-fade transitions
  and merges narration audio.
* **Blender** – uses the Blender Python API (``bpy``) for richer compositing
  (enabled via ``use_blender=True``).

Inputs
------
- scenes           : list[dict] – Scenes with ``clip_path`` and ``narration_audio_path``.
- output_path      : str | Path – Final video file path (e.g. ``output/final_output.mp4``).
- transition       : str        – ``"crossfade"``, ``"fade"``, or ``"cut"``.
- transition_dur   : float      – Transition duration in seconds.
- use_blender      : bool       – Use Blender compositing (default False).
- codec            : str        – FFmpeg codec (default ``"libx264"``).
- crf              : int        – Constant rate factor (default 18).

Outputs
-------
Path – The path to the assembled final video.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def assemble_video(
    scenes: list[dict[str, Any]],
    output_path: str | Path = "output/final_output.mp4",
    transition: str = "crossfade",
    transition_dur: float = 0.5,
    use_blender: bool = False,
    codec: str = "libx264",
    crf: int = 18,
) -> Path:
    """Combine per-scene clips + narration into one final video.

    Returns the *output_path* as a :class:`Path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_blender:
        return _assemble_blender(scenes, output_path)

    return _assemble_ffmpeg(scenes, output_path, transition, transition_dur, codec, crf)


# ──────────────────────────────────────────────────────────────────────
# FFmpeg assembly
# ──────────────────────────────────────────────────────────────────────

def _assemble_ffmpeg(
    scenes: list[dict[str, Any]],
    output_path: Path,
    transition: str,
    transition_dur: float,
    codec: str,
    crf: int,
) -> Path:
    """Concatenate clips with optional crossfade and merge narration."""
    clips = [Path(s["clip_path"]) for s in scenes]
    narrations = [Path(s["narration_audio_path"]) for s in scenes]

    if transition == "cut" or len(clips) < 2:
        merged_video = _concat_clips(clips)
    else:
        merged_video = _crossfade_clips(clips, transition_dur)

    merged_audio = _concat_narration(narrations)

    # Mux video + audio
    logger.info("Muxing video + audio → %s", output_path)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(merged_video),
            "-i", str(merged_audio),
            "-c:v", codec, "-crf", str(crf),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    logger.info("Final video saved: %s", output_path)
    return output_path


def _concat_clips(clips: list[Path]) -> Path:
    """Simple concat demuxer."""
    tmp = Path(tempfile.mktemp(suffix="_clips.txt"))
    tmp.write_text("\n".join(f"file '{Path(c).resolve()}'" for c in clips))
    out = clips[0].parent / "_merged_video.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp), "-c", "copy", str(out)],
        check=True, capture_output=True,
    )
    tmp.unlink(missing_ok=True)
    return out


def _crossfade_clips(clips: list[Path], dur: float) -> Path:
    """Build an FFmpeg xfade filter chain for crossfade transitions."""
    if len(clips) == 1:
        return clips[0]

    # Build inputs and xfade filter chain
    inputs: list[str] = []
    for c in clips:
        inputs.extend(["-i", str(c)])

    filter_parts: list[str] = []
    current = "[0:v]"
    for i in range(1, len(clips)):
        next_label = f"[v{i}]"
        offset = i * 5 - dur  # rough offset; in production, derive from duration_hint
        if offset < 0:
            offset = 0
        out_label = f"[xf{i}]" if i < len(clips) - 1 else "[outv]"
        filter_parts.append(
            f"{current}[{i}:v]xfade=transition=fade:duration={dur}:offset={offset}{out_label}"
        )
        current = out_label

    filter_str = ";".join(filter_parts)
    out = clips[0].parent / "_merged_video.mp4"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_str, "-map", "[outv]", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _concat_narration(narrations: list[Path]) -> Path:
    """Concatenate per-scene narration WAVs."""
    tmp = Path(tempfile.mktemp(suffix="_narr.txt"))
    tmp.write_text("\n".join(f"file '{Path(n).resolve()}'" for n in narrations))
    out = narrations[0].parent / "_merged_narration.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp), "-c", "copy", str(out)],
        check=True, capture_output=True,
    )
    tmp.unlink(missing_ok=True)
    return out


# ──────────────────────────────────────────────────────────────────────
# Blender assembly (optional)
# ──────────────────────────────────────────────────────────────────────

def _assemble_blender(scenes: list[dict[str, Any]], output_path: Path) -> Path:
    """Use Blender's Python API to composite clips + audio.

    Requires Blender to be importable (``import bpy``).
    """
    try:
        import bpy  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Blender Python module (bpy) is not available. "
            "Set use_blender=False or install Blender with Python support."
        ) from exc

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.fps = 24

    if not scene.sequence_editor:
        scene.sequence_editor_create()
    seq = scene.sequence_editor

    channel = 1
    frame_cursor = 1
    for s in scenes:
        clip = Path(s["clip_path"])
        audio = Path(s["narration_audio_path"])
        duration_frames = int(s.get("duration_hint", 5) * 24)

        seq.sequences.new_movie(clip.stem, str(clip), channel, frame_cursor)
        seq.sequences.new_sound(audio.stem, str(audio), channel + 1, frame_cursor)
        frame_cursor += duration_frames
        channel += 2

    scene.frame_end = frame_cursor
    bpy.ops.render.render(animation=True)
    logger.info("Blender render complete → %s", output_path)
    return output_path
