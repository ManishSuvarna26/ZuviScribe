"""
tools.narration_tts
~~~~~~~~~~~~~~~~~~~
Generate high-quality narration audio for each scene using Microsoft
Edge TTS (free, natural-sounding voices).

This is the timing backbone of the documentary pipeline  the TTS audio
duration for each scene determines how long that scene's video clip will be.

Inputs
------
- scenes     : list[dict] – Script scenes with ``narration`` text.
- output_dir : str | Path – Where to write MP3 files.
- voice      : str        – Edge TTS voice ID.

Outputs
-------
list[dict] – Scenes augmented with:
    narration_audio_path : Path  – Path to the generated audio file.
    narration_duration   : float – Exact duration in seconds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cinematic documentary voice  authoritative, clear, reliable
_DEFAULT_VOICE = "en-US-ChristopherNeural"  # Authority/News voice, ideal for docs


def generate_narration(
    scenes: list[dict[str, Any]],
    output_dir: str | Path = "output/narration",
    voice: str = _DEFAULT_VOICE,
    **_extra: Any,
) -> list[dict[str, Any]]:
    """Generate TTS audio for each scene's narration text.

    Returns scenes augmented with ``narration_audio_path`` and
    ``narration_duration`` (seconds).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for scene in scenes:
        num = scene["scene_number"]
        text = scene.get("narration", "").strip()
        mp3_path = output_dir / f"narration_{num:03d}.mp3"

        if not text:
            logger.warning("Scene %d has no narration text. Generating 3s silence.", num)
            _generate_silence(mp3_path, 3.0)
            results.append({**scene, "narration_audio_path": mp3_path, "narration_duration": 3.0})
            continue

        logger.info("Generating TTS for scene %d (%d chars) …", num, len(text))
        asyncio.run(_edge_tts_generate(text, voice, mp3_path))

        duration = _get_audio_duration(mp3_path)
        logger.info("Scene %d narration: %.1fs → %s", num, duration, mp3_path)

        results.append({
            **scene,
            "narration_audio_path": mp3_path,
            "narration_duration": duration,
        })

    total = sum(s["narration_duration"] for s in results)
    logger.info("Narration complete – %d scenes, total %.1fs.", len(results), total)
    return results


async def _edge_tts_generate(text: str, voice: str, output_path: Path) -> None:
    """Use edge-tts to generate an MP3 file with documentary pacing."""
    import edge_tts

    # Slightly slower rate for measured documentary delivery
    communicate = edge_tts.Communicate(text, voice, rate="-8%", pitch="-2Hz")
    try:
        await communicate.save(str(output_path))
    except Exception:
        # Fallback to default voice if the requested one fails
        communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
        await communicate.save(str(output_path))


def _get_audio_duration(path: Path) -> float:
    """Get audio duration via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 5.0


def _generate_silence(path: Path, duration: float) -> None:
    """Generate a silent MP3 of the given duration."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
            "-t", str(duration),
            "-c:a", "libmp3lame", "-q:a", "2",
            str(path),
        ],
        check=True, capture_output=True,
    )
