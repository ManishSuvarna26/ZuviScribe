"""
tools.transcript_refiner
~~~~~~~~~~~~~~~~~~~~~~~~
LLM-based post-processing of Whisper transcripts to fix ASR errors.

Whisper (especially the tiny/base models) makes systematic errors on
technical and mathematical content: mishearing domain terms, losing
proper nouns, confusing variable names, etc.  This module sends the raw
transcript through an Ollama LLM to correct those errors while
preserving timestamps and structure.

Inputs
------
- transcript : dict  – From transcriber (full_text, segments, language).
- model      : str   – Ollama model name (default: gemma2:27b).
- host       : str   – Ollama API base URL.

Outputs
-------
dict – Same structure as input, with corrected text in segments and full_text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a transcript correction assistant. You will receive a raw \
automatic speech recognition (ASR) transcript of an academic / technical \
lecture. The ASR system makes mistakes, especially on domain-specific \
vocabulary, mathematical terminology, and variable names.

Your task:
1. Fix misheard words and phrases (e.g. "sense of fusion" → "sensor fusion", \
"the noted" → "denoted", "p and f" → "PMF").
2. Correct mathematical or variable references spoken aloud \
(e.g. "capital P of said" → "P(x)", "lowercase p of said" → "p(x)", \
"said" when referring to a random variable → the correct variable like "x" or "z").
3. Fix any obvious grammatical errors introduced by ASR.
4. Preserve the original meaning and sentence structure as much as possible. \
Do NOT paraphrase or rewrite content  only fix errors.
5. Do NOT add any commentary, explanations, or formatting. Return ONLY the \
corrected transcript text.

Return ONLY a JSON array of corrected segment texts, one string per input segment, \
in the same order. The array MUST have EXACTLY the same number of elements as the \
input array  one corrected string per input string. No markdown fences, no \
explanation  just the JSON array."""

CHUNK_SIZE = 10  # segments per LLM call (small for reliable JSON output)


def refine_transcript(
    transcript: dict[str, Any],
    model: str = "gemma2:27b",
    host: str = "http://localhost:11434",
    progress: Any = None,
) -> dict[str, Any]:
    """Refine Whisper transcript using an Ollama LLM."""
    segments = transcript.get("segments", [])
    if not segments:
        logger.warning("No segments to refine.")
        return transcript

    total_chunks = (len(segments) + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info(
        "Refining %d transcript segments with %s (%d chunks) …",
        len(segments), model, total_chunks,
    )

    # Add a sub-task for chunk progress if a rich Progress is provided
    chunk_task = None
    if progress is not None:
        chunk_task = progress.add_task(
            "[3/4] LLM refining chunks", total=total_chunks,
        )

    refined_segments = []
    for i in range(0, len(segments), CHUNK_SIZE):
        chunk = segments[i : i + CHUNK_SIZE]
        corrected_texts = _refine_chunk(chunk, model, host)

        for seg, new_text in zip(chunk, corrected_texts):
            refined_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": new_text,
            })

        if chunk_task is not None:
            progress.advance(chunk_task)

    full_text = " ".join(seg["text"] for seg in refined_segments)

    logger.info("Transcript refinement complete  %d segments processed.", len(refined_segments))
    return {
        "full_text": full_text,
        "segments": refined_segments,
        "language": transcript.get("language", "en"),
    }


def _refine_chunk(
    segments: list[dict],
    model: str,
    host: str,
) -> list[str]:
    """Send a chunk of segments to the LLM and return corrected texts."""
    texts = [seg.get("text", "") for seg in segments]

    user_message = (
        f"Here are {len(texts)} transcript segments to correct. "
        f"Return a JSON array with EXACTLY {len(texts)} corrected strings "
        f"(one per input segment, same order).\n\n"
        + json.dumps(texts, ensure_ascii=False)
    )

    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 4096},
            },
            timeout=300,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        corrected = _parse_json_array(content, len(texts), texts)
        return corrected

    except (requests.RequestException, ValueError) as exc:
        logger.warning("LLM refinement failed for chunk (%s). Keeping original text.", exc)
        return texts


def _parse_json_array(content: str, expected_len: int, originals: list[str]) -> list[str]:
    """Extract a JSON array of strings from the LLM response."""
    # Strip markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content.strip())

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to find a JSON array in the response
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse JSON array from LLM response: {content[:200]}")

    if not isinstance(result, list):
        raise ValueError(f"Expected JSON array, got {type(result).__name__}")

    if len(result) != expected_len:
        logger.warning(
            "LLM returned %d items but expected %d. Using originals for missing.",
            len(result), expected_len,
        )
        # Pad with original text for missing entries
        if len(result) < expected_len:
            result.extend(originals[len(result):])
        result = result[:expected_len]

    return [str(item) for item in result]
