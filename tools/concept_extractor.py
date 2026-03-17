"""
tools.concept_extractor
~~~~~~~~~~~~~~~~~~~~~~~
Extract visual concepts / scene descriptions from video frames using a
vision-language model (e.g. LLaVA) served by Ollama.

Inputs
------
- frame_paths : list[Path] – Ordered list of JPEG frame paths.
- model       : str        – Ollama vision model tag (default ``"llava"``).
- ollama_host : str        – Ollama server URL.
- max_concepts_per_frame : int – Cap the number of concepts per frame.

Outputs
-------
list[dict] – One dict per frame:
    frame_index : int
    path        : Path
    description : str   – Natural-language scene description.
    concepts    : list[str] – Key visual concepts (e.g. "desert", "car").
"""

from __future__ import annotations

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPT = (
    "Describe this video frame in one short paragraph. "
    "Then list up to {max_concepts} key visual concepts as a comma-separated list "
    "on a new line prefixed with 'Concepts:'. "
    "Be specific about colours, objects, actions, and setting."
)


def _encode_image(path: Path) -> str:
    """Return base64-encoded image bytes."""
    return base64.b64encode(path.read_bytes()).decode()


def _parse_response(text: str) -> tuple[str, list[str]]:
    """Split the model response into description + concept list."""
    lines = text.strip().split("\n")
    concepts: list[str] = []
    desc_lines: list[str] = []
    for line in lines:
        if line.lower().startswith("concepts:"):
            raw = line.split(":", 1)[1]
            concepts = [c.strip() for c in raw.split(",") if c.strip()]
        else:
            desc_lines.append(line)
    return "\n".join(desc_lines).strip(), concepts


def _probe_vision_support(client: Any, model: str) -> bool:
    """Quick test if the model actually supports image inputs."""
    try:
        info = client.show(model)
        # Ollama returns a Pydantic object  use attribute access
        details = getattr(info, "details", None)
        if details is None:
            return False
        families = getattr(details, "families", None) or []
        if any(f in families for f in ("clip", "llava", "vision")):
            return True
        family = getattr(details, "family", "")
        if family in ("llava", "llava-next", "minicpm-v", "moondream"):
            return True
    except Exception:
        pass
    return False


def extract_concepts(
    frame_paths: list[Path],
    model: str = "llava",
    ollama_host: str = "http://localhost:11434",
    max_concepts_per_frame: int = 5,
    max_frames_to_analyse: int = 15,
) -> list[dict[str, Any]]:
    """Analyse each frame with the Ollama vision model and return descriptions.

    If there are more frames than *max_frames_to_analyse*, evenly subsample
    to keep analysis time reasonable.  Frames not analysed get placeholder text.

    If the model does not support vision (or is unavailable), falls back to
    returning lightweight placeholder descriptions so the pipeline continues.
    """
    import ollama as _ollama

    client = _ollama.Client(host=ollama_host)
    prompt = _PROMPT.format(max_concepts=max_concepts_per_frame)
    results: list[dict[str, Any]] = []

    # Probe once: skip expensive per-frame API calls for non-vision models
    use_vision = _probe_vision_support(client, model)
    if not use_vision:
        logger.warning(
            "Model '%s' does not appear to support vision. "
            "Skipping per-frame API calls; using placeholder descriptions.",
            model,
        )
        for idx, fpath in enumerate(frame_paths):
            results.append(
                {
                    "frame_index": idx,
                    "path": Path(fpath),
                    "description": f"Scene {idx + 1} from the original video.",
                    "concepts": [],
                }
            )
        logger.info("Concept extraction complete – %d frames (placeholders).", len(results))
        return results

    # Subsample if too many frames
    n = len(frame_paths)
    if n > max_frames_to_analyse:
        step = n / max_frames_to_analyse
        analyse_indices = {int(i * step) for i in range(max_frames_to_analyse)}
        logger.info("Subsampling %d frames → %d for vision analysis.", n, len(analyse_indices))
    else:
        analyse_indices = set(range(n))

    for idx, fpath in enumerate(frame_paths):
        fpath = Path(fpath)

        # Skip frames not in the subsample
        if idx not in analyse_indices:
            results.append(
                {
                    "frame_index": idx,
                    "path": fpath,
                    "description": f"Scene {idx + 1} from the original video.",
                    "concepts": [],
                }
            )
            continue

        logger.info("Analysing frame %d / %d – %s", idx + 1, len(frame_paths), fpath.name)

        description = ""
        concepts: list[str] = []
        try:
            def _call_llava():
                return client.chat(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [_encode_image(fpath)],
                        }
                    ],
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_llava)
                resp = future.result(timeout=60)  # 60s per frame

            # Ollama returns Pydantic objects  handle both attribute and dict access
            msg = getattr(resp, "message", None) or resp["message"]
            text = getattr(msg, "content", None) or msg["content"]
            description, concepts = _parse_response(text)
        except FuturesTimeout:
            logger.warning(
                "Frame %d timed out after 60s. Using placeholder.", idx,
            )
            description = f"Scene {idx + 1} from the original video."
            concepts = []
        except Exception as exc:
            logger.warning(
                "Vision model '%s' unavailable for frame %d (%s). "
                "Using placeholder description.",
                model, idx, exc,
            )
            description = f"Scene {idx + 1} from the original video."
            concepts = []

        results.append(
            {
                "frame_index": idx,
                "path": fpath,
                "description": description,
                "concepts": concepts[:max_concepts_per_frame],
            }
        )

    logger.info("Concept extraction complete – %d frames processed.", len(results))
    return results
