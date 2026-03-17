"""
tools.documentary_assembler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assemble animated clips + narration audio into a polished documentary video.

Features:
- Crossfade transitions between scenes
- Narration audio perfectly synced to each clip
- Optional background music at low volume under narration
- Professional encoding settings

Inputs
------
- scenes      : list[dict] – Scenes with ``clip_path``, ``narration_audio_path``.
- output_path : str | Path – Final video file.
- transition  : str        – "crossfade" | "cut"
- transition_dur : float   – Crossfade duration in seconds.

Outputs
-------
Path – The assembled video file.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def assemble_documentary(
    scenes: list[dict[str, Any]],
    output_path: str | Path = "output/final_output.mp4",
    transition: str = "crossfade",
    transition_dur: float = 0.8,
    codec: str = "libx264",
    crf: int = 18,
    music_path: str | Path | None = None,
    music_volume: float = 0.12,
    **_extra: Any,
) -> Path:
    """Combine per-scene clips + narration + optional background music."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clips = [Path(s["clip_path"]) for s in scenes]
    narrations = [Path(s["narration_audio_path"]) for s in scenes]

    if len(clips) < 2 or transition == "cut":
        merged_video = _concat_clips(clips)
    else:
        merged_video = _crossfade_clips(clips, scenes, transition_dur)

    merged_audio = _concat_audio(narrations)

    # Mix background music under narration if provided
    if music_path and Path(music_path).exists():
        logger.info("Mixing background music (volume=%.0f%%) with narration …", music_volume * 100)
        final_audio = _mix_audio_with_music(merged_audio, Path(music_path), music_volume)
    else:
        final_audio = merged_audio

    # Mux video + audio with professional settings
    logger.info("Muxing video + audio → %s", output_path)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(merged_video),
            "-i", str(final_audio),
            "-c:v", codec, "-crf", str(crf),
            "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ],
        check=True, capture_output=True,
    )

    logger.info("Final documentary saved: %s", output_path)
    return output_path


def _concat_clips(clips: list[Path]) -> Path:
    """Simple concat via demuxer."""
    tmp = Path(tempfile.mktemp(suffix="_clips.txt"))
    tmp.write_text("\n".join(f"file '{c.resolve()}'" for c in clips))
    out = clips[0].parent / "_merged_video.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp), "-c", "copy", str(out)],
        check=True, capture_output=True,
    )
    tmp.unlink(missing_ok=True)
    return out


def _crossfade_clips(clips: list[Path], scenes: list[dict], dur: float) -> Path:
    """Build crossfade transitions between clips using xfade filter."""
    if len(clips) == 1:
        return clips[0]

    # Get actual clip durations
    durations = []
    for i, scene in enumerate(scenes):
        d = _get_video_duration(clips[i])
        if d <= 0:
            d = scene.get("narration_duration", 10.0)
        durations.append(d)

    # Build inputs
    inputs: list[str] = []
    for c in clips:
        inputs.extend(["-i", str(c)])

    # Build xfade filter chain
    filter_parts: list[str] = []
    current = "[0:v]"
    cumulative_offset = 0.0

    for i in range(1, len(clips)):
        cumulative_offset += durations[i - 1] - dur
        offset = max(cumulative_offset, 0)
        out_label = f"[xf{i}]" if i < len(clips) - 1 else "[outv]"
        filter_parts.append(
            f"{current}[{i}:v]xfade=transition=fade:duration={dur}:offset={offset:.3f}{out_label}"
        )
        current = out_label

    filter_str = ";".join(filter_parts)
    out = clips[0].parent / "_merged_video.mp4"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(out),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("Crossfade failed (%s), falling back to concat.", exc.stderr[-200:] if exc.stderr else exc)
        return _concat_clips(clips)

    return out


def _concat_audio(paths: list[Path]) -> Path:
    """Concatenate narration audio files."""
    tmp = Path(tempfile.mktemp(suffix="_narr.txt"))
    tmp.write_text("\n".join(f"file '{p.resolve()}'" for p in paths))
    out = paths[0].parent / "_merged_narration.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp), "-c", "copy", str(out)],
        check=True, capture_output=True,
    )
    tmp.unlink(missing_ok=True)
    return out


def _mix_audio_with_music(
    narration: Path, music: Path, music_vol: float,
) -> Path:
    """Mix narration with background music at reduced volume."""
    out = narration.parent / "_mixed_audio.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(narration),
            "-i", str(music),
            "-filter_complex",
            f"[1:a]volume={music_vol}[m];"
            f"[0:a][m]amix=inputs=2:duration=first:normalize=0[out]",
            "-map", "[out]",
            "-c:a", "libmp3lame", "-q:a", "2",
            str(out),
        ],
        check=True, capture_output=True,
    )
    logger.info("Audio mixed: narration + music (%.0f%% vol)", music_vol * 100)
    return out


def _get_video_duration(path: Path) -> float:
    """Get duration via ffprobe."""
    try:
        import json
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        return 0.0
