"""
tools.script_writer
~~~~~~~~~~~~~~~~~~~
Generate a documentary narration script from a video transcript.

Uses a creative LLM to transform dry educational/informational content into
an engaging, documentary-style narration in the vein of channels like
Fern or LEMMiNO  thoughtful pacing, compelling language, vivid imagery
descriptions suitable for cinematic visuals.

The output is a list of "beats"  each with narration text AND a detailed
visual description for image generation.

Inputs
------
- full_text   : str          – Full transcript of the original video.
- concepts    : list[dict]   – Visual concepts extracted from frames.
- model       : str          – Ollama text model tag.
- ollama_host : str          – Ollama server URL.

Outputs
-------
list[dict] – Ordered list of scene dicts:
    scene_number      : int
    narration         : str    – What the narrator says (for TTS).
    visual_prompt     : str    – Detailed image prompt (for SDXL).
    mood              : str    – Color/lighting mood keyword.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a world-class documentary scriptwriter for channels like \
Kurzgesagt, Fern, and LEMMiNO. Transform the transcript into a \
captivating visual documentary script.

RULES:
1. Rewrite content as a compelling narrator monologue. Do NOT repeat the \
   transcript verbatim  create vivid analogies, metaphors, and story \
   structure: hook → build → climax → resolution.
2. Create {min_scenes}–{max_scenes} scenes.
3. Each scene has:
   - "narration": 2-3 punchy sentences. Every word must earn its place. \
     Write for a professional voice actor performing a high-budget \
     documentary. Use rhetorical questions, dramatic pauses (ellipsis), \
     and powerful imagery.
   - "visual_prompt": Under 45 words. A cinematic image description for \
     Stable Diffusion XL. Include: main subject, composition, lighting, \
     color palette, camera angle. Focus on: dark background, luminous \
     subjects, volumetric lighting, hyper-detailed 3D render, minimalist \
     infographic. NEVER include text, words, letters, or UI elements.
   - "mood": one of "epic", "contemplative", "dramatic", "warm", \
     "mysterious", "energetic", "serene"
4. Total narration at 150 wpm should be 90–150 seconds. Be concise.
5. Open with a HOOK  a surprising fact, paradox, or provocative question.
6. End with a memorable conclusion that ties back to the opening hook.

Return ONLY a JSON array, no markdown fencing, no extra text:
[{{"scene_number":1,"narration":"...","visual_prompt":"...","mood":"epic"}}, ...]
"""


def write_script(
    full_text: str,
    concepts: list[dict[str, Any]],
    model: str = "gemma2:27b",
    ollama_host: str = "http://localhost:11434",
    min_scenes: int = 6,
    max_scenes: int = 12,
    **_extra: Any,
) -> list[dict[str, Any]]:
    """Generate a documentary narration script from a transcript."""
    import ollama as _ollama

    client = _ollama.Client(host=ollama_host)

    concept_summary = "\n".join(
        f"- Frame {c['frame_index']}: {c['description']}"
        for c in concepts
        if c.get("description", "").strip()
        and not c["description"].startswith("Scene ")
    )

    system = _SYSTEM_PROMPT.format(
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )

    user_message = (
        f"## Original Transcript\n{full_text}\n\n"
        f"## Visual Context from Original Footage\n{concept_summary}\n\n"
        f"Write the documentary script now."
    )

    logger.info("Generating documentary script with model=%s …", model)
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )

    msg = getattr(resp, "message", None) or resp["message"]
    raw: str = getattr(msg, "content", None) or msg["content"]
    scenes = _parse_script(raw)

    # Clamp to max_scenes
    scenes = scenes[:max_scenes]

    logger.info("Documentary script ready – %d scenes.", len(scenes))
    return scenes


def _parse_script(raw: str) -> list[dict[str, Any]]:
    """Best-effort JSON extraction from LLM output."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    scenes = None

    # Strategy 1: direct parse
    try:
        scenes = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: JSONDecoder.raw_decode from first '['
    if scenes is None:
        start = cleaned.find("[")
        if start != -1:
            try:
                scenes, _ = json.JSONDecoder().raw_decode(cleaned, start)
            except json.JSONDecodeError:
                pass

    # Strategy 3: regex individual objects
    if scenes is None:
        objects = re.findall(r'\{[^{}]*\}', cleaned)
        if objects:
            parsed = []
            for obj in objects:
                try:
                    parsed.append(json.loads(obj))
                except json.JSONDecodeError:
                    continue
            if parsed:
                scenes = parsed

    if not scenes or not isinstance(scenes, list):
        raise ValueError(f"Could not parse script JSON from LLM. Raw: {raw[:500]}")

    validated: list[dict[str, Any]] = []
    for i, s in enumerate(scenes, start=1):
        validated.append({
            "scene_number": s.get("scene_number", i),
            "narration": s.get("narration", ""),
            "visual_prompt": s.get("visual_prompt", s.get("description", "")),
            "mood": s.get("mood", "contemplative"),
        })

    return validated
