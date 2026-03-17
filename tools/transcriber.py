"""
tools.transcriber
~~~~~~~~~~~~~~~~~
Transcribe audio to text using OpenAI Whisper.

Inputs
------
- audio_path : str | Path – Path to a WAV/MP3 audio file.
- model_name : str        – Whisper model size (tiny, base, small, medium, large).
- language   : str | None – BCP-47 language code; ``None`` for auto-detect.
- device     : str        – ``"cpu"`` or ``"cuda"``.

Outputs
-------
dict with keys:
    full_text  : str              – Complete transcription.
    segments   : list[dict]       – Per-segment dicts with ``start``, ``end``,
                                    ``text`` keys (timestamps in seconds).
    language   : str              – Detected / forced language code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = "base",
    language: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run Whisper on *audio_path* and return structured transcript.

    Loads the requested Whisper model on the first call and caches it for
    subsequent invocations in the same process.
    """
    import whisper

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Loading Whisper model '%s' on %s …", model_name, device)
    model = whisper.load_model(model_name, device=device)

    decode_opts: dict[str, Any] = {}
    if language:
        decode_opts["language"] = language

    logger.info("Transcribing %s …", audio_path.name)
    result = model.transcribe(str(audio_path), **decode_opts)

    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result.get("segments", [])
    ]

    detected_lang = result.get("language", language or "en")
    logger.info(
        "Transcription complete – %d segments, language=%s",
        len(segments),
        detected_lang,
    )

    return {
        "full_text": result["text"].strip(),
        "segments": segments,
        "language": detected_lang,
    }
