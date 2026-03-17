"""
tools.audio_synth
~~~~~~~~~~~~~~~~~
Synthesize narration audio from the script, or remix the original audio.

Inputs
------
- scenes       : list[dict]   – Storyboard scenes (need ``narration`` text).
- output_dir   : str | Path   – Where to write WAV files.
- engine       : str          – ``"coqui"`` (TTS) or ``"original"`` (copy audio).
- original_audio : Path | None – Path to the original extracted audio.
- coqui_model  : str          – Coqui-TTS model identifier.
- sample_rate  : int          – Target sample rate.

Outputs
-------
list[dict] – Scenes augmented with ``narration_audio_path`` (Path).
Also returns a combined ``full_narration.wav`` covering all scenes.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def synthesize_audio(
    scenes: list[dict[str, Any]],
    output_dir: str | Path = "output/audio",
    engine: str = "coqui",
    original_audio: Path | None = None,
    audio_path: Path | None = None,  # alias forwarded by orchestrator
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC",
    sample_rate: int = 22050,
) -> list[dict[str, Any]]:
    """Generate narration WAVs for each scene.

    When *engine* is ``"original"`` and a source audio file is provided (via
    *original_audio* or *audio_path*), the original audio is split into equal
    segments matching each scene's ``duration_hint``.  Otherwise Coqui-TTS
    generates fresh speech.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accept either kwarg name for the source audio
    source = original_audio or audio_path

    if engine == "original" and source and Path(source).exists():
        return _split_original(scenes, Path(source), output_dir)

    return _tts_generate(scenes, output_dir, coqui_model, sample_rate)


def _tts_generate(
    scenes: list[dict[str, Any]],
    output_dir: Path,
    model_name: str,
    sample_rate: int,
) -> list[dict[str, Any]]:
    """Use Coqui-TTS to synthesize narration for each scene."""
    from TTS.api import TTS

    logger.info("Loading Coqui-TTS model: %s", model_name)
    tts = TTS(model_name=model_name)

    results: list[dict[str, Any]] = []
    for scene in scenes:
        num = scene["scene_number"]
        text = scene.get("narration", "")
        wav_path = output_dir / f"narration_{num:03d}.wav"

        if text.strip():
            logger.info("Synthesizing narration for scene %d …", num)
            tts.tts_to_file(text=text, file_path=str(wav_path))
        else:
            # Generate a silent WAV placeholder
            duration = scene.get("duration_hint", 5.0)
            _generate_silence(wav_path, duration, sample_rate)

        results.append({**scene, "narration_audio_path": wav_path})

    # Concatenate all narrations
    _concat_audio([r["narration_audio_path"] for r in results], output_dir / "full_narration.wav")
    logger.info("Audio synthesis complete – %d segments.", len(results))
    return results


def _split_original(
    scenes: list[dict[str, Any]],
    audio_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Split the original audio into per-scene segments based on duration_hint."""
    results: list[dict[str, Any]] = []
    offset = 0.0
    for scene in scenes:
        num = scene["scene_number"]
        dur = scene.get("duration_hint", 10.0)
        seg_path = output_dir / f"narration_{num:03d}.wav"

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-ss", str(offset),
                "-t", str(dur),
                "-acodec", "pcm_s16le",
                str(seg_path),
            ],
            check=True,
            capture_output=True,
        )
        offset += dur
        results.append({**scene, "narration_audio_path": seg_path})

    _concat_audio([r["narration_audio_path"] for r in results], output_dir / "full_narration.wav")
    return results


def _generate_silence(path: Path, duration: float, sample_rate: int) -> None:
    """Write a silent WAV file of the given duration."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", str(duration),
            "-acodec", "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _concat_audio(paths: list[Path], output: Path) -> None:
    """Concatenate multiple WAV files into a single file using FFmpeg."""
    if not paths:
        return
    list_file = output.parent / "_concat_list.txt"
    list_file.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in paths))
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    list_file.unlink(missing_ok=True)
