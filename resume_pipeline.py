#!/usr/bin/env python3
"""
resume_pipeline.py – Resume the pipeline from after download+transcription.

Reconstructs the context from existing artifacts on disk and runs the
remaining steps: concept extraction → storyboard → render → animate →
audio → assemble.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("resume")

OUTPUT_DIR = Path("output")
DOWNLOAD_DIR = OUTPUT_DIR / "download"


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f) or {}


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


def reconstruct_context() -> dict:
    """Rebuild the context dict from files already on disk."""
    # Find the downloaded video
    videos = list(DOWNLOAD_DIR.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError("No .mp4 found in output/download/")
    video = videos[0]
    title = video.stem.replace(" ", "_")

    # Get video duration
    duration = _get_video_duration(video)
    logger.info("Original video duration: %.1f seconds", duration)

    # Audio
    audio_files = list(DOWNLOAD_DIR.glob("*_audio.wav"))
    if not audio_files:
        raise FileNotFoundError("No audio WAV found in output/download/")
    audio_path = audio_files[0]

    # Frames
    frames_dir = DOWNLOAD_DIR / "frames"
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    logger.info("Found %d frames in %s", len(frame_paths), frames_dir)

    # Re-run transcription to get full_text (fast with tiny model)
    logger.info("Re-running Whisper transcription (tiny model, fast)...")
    from tools.transcriber import transcribe_audio
    transcript = transcribe_audio(audio_path=audio_path, model_name="tiny", device="cpu")

    context = {
        "url": "https://www.youtube.com/watch?v=ijX8Xs0OaAs",
        "title": title,
        "audio_path": audio_path,
        "frame_paths": frame_paths,
        "metadata": {"url": "resumed", "duration": duration},
        "full_text": transcript["full_text"],
        "segments": transcript["segments"],
        "language": transcript["language"],
    }
    logger.info("Context reconstructed. Title: %s, Frames: %d, Duration: %.1fs",
                title, len(frame_paths), duration)
    return context


async def run_remaining_steps(context: dict, config: dict) -> dict:
    """Run steps from concept extraction through final assembly."""
    from agents.pipeline import build_pipeline
    from agents.orchestrator import Orchestrator

    pipeline = build_pipeline()

    # Skip download, transcribe, and upload
    skip = {"download", "transcribe", "upload"}
    remaining = [s for s in pipeline if s.name not in skip]

    orch = Orchestrator(
        url=context["url"],
        config=config,
        dry_run=False,
        upload=False,
        output_dir=str(OUTPUT_DIR),
    )
    orch.context = context

    for step in remaining:
        logger.info("▶ Running step: %s – %s", step.name, step.description)
        try:
            await orch._execute_step(step)
            logger.info("✔ %s complete.", step.name)
        except Exception as exc:
            logger.error("✖ %s FAILED: %s", step.name, exc, exc_info=True)
            raise

    return orch.context


def main():
    config = load_config()
    context = reconstruct_context()
    result = asyncio.run(run_remaining_steps(context, config))

    if "output_path" in result:
        logger.info("🎬 Final video: %s", result["output_path"])
    else:
        logger.error("No output_path in final context – pipeline may have failed.")
        logger.info("Final context keys: %s", list(result.keys()))


if __name__ == "__main__":
    main()
