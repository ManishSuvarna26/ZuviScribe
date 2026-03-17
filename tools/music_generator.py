"""
tools.music_generator
~~~~~~~~~~~~~~~~~~~~~
Generate ambient background music for documentary narration.

Primary: MusicGen-small from Meta (AI-generated contextual music).
Fallback: FFmpeg-synthesized ambient drone (no download needed).

Inputs
------
- total_duration : float – Required music duration in seconds.
- output_path    : str   – Where to save the music file (WAV).
- mood           : str   – Overall mood for the music.

Outputs
-------
Path – The generated music file.
"""

from __future__ import annotations

import logging
import subprocess
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_background_music(
    total_duration: float,
    output_path: str | Path = "output/background_music.wav",
    mood: str = "cinematic",
    **_extra: Any,
) -> Path:
    """Generate ambient background music."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return _generate_musicgen(total_duration, output_path, mood)
    except Exception as exc:
        logger.warning("MusicGen unavailable (%s), using FFmpeg ambient fallback.", exc)
        return _generate_ambient_fallback(total_duration, output_path)


def _generate_musicgen(total_duration: float, output_path: Path, mood: str) -> Path:
    """Generate music with facebook/musicgen-small."""
    import numpy as np
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    logger.info("Loading MusicGen-small …")

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    # CPU is fast enough for musicgen-small and avoids MPS issues
    device = "cpu"
    model = model.to(device)

    mood_prompts = {
        "epic": "epic cinematic orchestral music, dramatic strings, slow crescendo, film score",
        "contemplative": "soft ambient piano, gentle atmospheric pads, contemplative, minimal",
        "dramatic": "tense cinematic underscore, dark strings, pulsing bass, suspenseful",
        "warm": "warm acoustic guitar background, soft strings, hopeful gentle melody",
        "mysterious": "dark ambient drone, eerie pads, subtle dissonance, mysterious atmosphere",
        "energetic": "upbeat electronic music, driving synth, dynamic documentary score",
        "serene": "peaceful ambient piano, nature-inspired, calming pads, gentle melody",
    }

    prompt = mood_prompts.get(mood,
        "ambient cinematic documentary background music, subtle, atmospheric, slow")

    logger.info("Generating background music: '%s' …", prompt[:60])

    # Generate ~30 seconds (each token ≈ 20ms at 32kHz)
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=1500)

    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_np = audio_values[0, 0].cpu().numpy()

    # Normalize
    peak = max(abs(audio_np.max()), abs(audio_np.min()), 1e-8)
    audio_np = audio_np / peak

    # Save 30s segment as WAV using stdlib wave module
    temp_wav = output_path.with_suffix(".tmp.wav")
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(temp_wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(audio_int16.tobytes())

    # Loop to match total duration
    _loop_audio(temp_wav, output_path, total_duration)
    temp_wav.unlink(missing_ok=True)

    # Free memory
    del model, processor
    import gc
    gc.collect()

    logger.info("Background music generated: %s (%.1fs)", output_path, total_duration)
    return output_path


def _generate_ambient_fallback(total_duration: float, output_path: Path) -> Path:
    """Generate layered ambient drone using FFmpeg synthesis."""
    logger.info("Generating ambient soundscape with FFmpeg (%.1fs) …", total_duration)

    dur = str(total_duration)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=65:duration={dur}",
            "-f", "lavfi", "-i", f"sine=frequency=98:duration={dur}",
            "-f", "lavfi", "-i", f"sine=frequency=131:duration={dur}",
            "-filter_complex",
            "[0:a]volume=0.15[a];[1:a]volume=0.08[b];[2:a]volume=0.04[c];"
            "[a][b][c]amix=inputs=3:normalize=0[mix];"
            "[mix]lowpass=f=300,tremolo=f=0.07:d=0.4,volume=0.5[out]",
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            str(output_path),
        ],
        check=True, capture_output=True,
    )
    logger.info("Ambient soundtrack generated: %s", output_path)
    return output_path


def _loop_audio(source: Path, output: Path, target_duration: float) -> None:
    """Loop audio file to fill target duration."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", str(source),
            "-t", str(target_duration),
            "-c:a", "pcm_s16le",
            str(output),
        ],
        check=True, capture_output=True,
    )
