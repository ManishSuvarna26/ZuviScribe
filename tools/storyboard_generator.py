"""
tools.storyboard_generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a creative animated storyboard from visual concepts and a transcript.

The LLM is instructed to preserve the original plot while transforming the
visual style into an imaginative, animated rendition.

Inputs
------
- transcript    : str          – Full transcription text.
- concepts      : list[dict]   – Output of ``concept_extractor.extract_concepts``.
- model         : str          – Ollama text model tag.
- ollama_host   : str          – Ollama server URL.
- min_scenes    : int          – Minimum scene count (default 5).
- max_scenes    : int          – Maximum scene count (default 10).
- style         : str          – Desired animation style prompt fragment.

Outputs
-------
list[dict] – One dict per scene:
    scene_number  : int
    description   : str   – Detailed visual description for image generation.
    narration     : str   – Narration text for this scene.
    duration_hint : float – Suggested duration in seconds.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a creative storyboard artist. Given a video transcript and a list of
visual scene descriptions extracted from the original footage, produce a NEW
storyboard that:

1. PRESERVES the original storyline and key plot points exactly.
2. TRANSFORMS the visual style into: {style}.
3. Contains between {min_scenes} and {max_scenes} scenes.
4. Each scene has a rich visual description suitable as a Stable Diffusion prompt,
   a narration passage, and a suggested duration in seconds.
5. The sum of all scene durations MUST equal approximately {total_duration:.0f} seconds.
   Distribute durations proportionally based on how much content each scene covers.

Return ONLY a JSON array (no markdown fencing) where each element is an object:
{{"scene_number": 1, "description": "...", "narration": "...", "duration_hint": 12.5}}
"""


def generate_storyboard(
    full_text: str,
    concepts: list[dict[str, Any]],
    model: str = "qwen3.5:latest",
    ollama_host: str = "http://localhost:11434",
    min_scenes: int = 5,
    max_scenes: int = 10,
    style: str = "vibrant animated cartoon",
    video_duration: float = 0.0,
) -> list[dict[str, Any]]:
    """Ask an LLM to produce a style-shifted storyboard from the original content.

    Returns a list of scene dicts ready for the image renderer.
    """
    import ollama as _ollama

    client = _ollama.Client(host=ollama_host)

    # Default to a reasonable duration if not provided
    if video_duration <= 0:
        video_duration = 120.0

    concept_summary = "\n".join(
        f"- Frame {c['frame_index']}: {c['description']}" for c in concepts
    )

    system = _SYSTEM_PROMPT.format(
        style=style,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
        total_duration=video_duration,
    )

    user_message = (
        f"## Original Transcript\n{full_text}\n\n"
        f"## Visual Scene Descriptions\n{concept_summary}\n\n"
        f"Generate the storyboard now."
    )

    logger.info("Generating storyboard with model=%s …", model)
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )

    # Ollama returns Pydantic objects  handle both attribute and dict access
    msg = getattr(resp, "message", None) or resp["message"]
    raw: str = getattr(msg, "content", None) or msg["content"]
    scenes = _parse_storyboard(raw, min_scenes, max_scenes)

    # Normalise durations so they sum to the original video duration
    total_raw = sum(s["duration_hint"] for s in scenes)
    if total_raw > 0 and video_duration > 0:
        scale = video_duration / total_raw
        for s in scenes:
            s["duration_hint"] = round(s["duration_hint"] * scale, 2)

    logger.info("Storyboard generated – %d scenes, total %.1fs.", len(scenes),
                sum(s["duration_hint"] for s in scenes))
    return scenes


def _parse_storyboard(
    raw: str, min_scenes: int, max_scenes: int
) -> list[dict[str, Any]]:
    """Best-effort JSON extraction from LLM output."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    scenes = None

    # Strategy 1: direct parse
    try:
        scenes = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: use JSONDecoder to parse just the first JSON value
    if scenes is None:
        # Find the first '[' and parse from there
        start = cleaned.find("[")
        if start != -1:
            decoder = json.JSONDecoder()
            try:
                scenes, _ = decoder.raw_decode(cleaned, start)
            except json.JSONDecodeError:
                pass

    # Strategy 3: extract individual {...} objects with regex
    if scenes is None:
        objects = re.findall(r'\{[^{}]*\}', cleaned)
        if objects:
            scenes = [json.loads(obj) for obj in objects]

    if scenes is None:
        raise ValueError("Could not parse storyboard JSON from LLM response")

    if not isinstance(scenes, list):
        raise TypeError("Expected a JSON array of scenes")

    # Validate and normalise
    validated: list[dict[str, Any]] = []
    for i, scene in enumerate(scenes[:max_scenes], start=1):
        validated.append(
            {
                "scene_number": scene.get("scene_number", i),
                "description": scene.get("description", ""),
                "narration": scene.get("narration", ""),
                "duration_hint": float(scene.get("duration_hint", 10.0)),
            }
        )

    if len(validated) < min_scenes:
        logger.warning(
            "LLM returned only %d scenes (minimum %d). Using what we have.",
            len(validated),
            min_scenes,
        )

    return validated
