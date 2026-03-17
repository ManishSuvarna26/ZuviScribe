#!/usr/bin/env python3
"""
main.py – Entry point for ZuviScribe.

Usage
-----
    # Animate a video (documentary-style)
    python main.py https://www.youtube.com/watch?v=EXAMPLE --animate

    # Transcribe a video to text file
    python main.py https://www.youtube.com/watch?v=EXAMPLE --transcribe

    # Both
    python main.py https://www.youtube.com/watch?v=EXAMPLE --animate --transcribe

    # With options
    python main.py --config my_config.yaml --output results/ URL --animate
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zuviscribe")


def _load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _ensure_downloaded(url: str, config: dict, output_dir: Path) -> dict:
    """Download video if not already present, return context dict."""
    from tools.downloader import download_video

    dl_dir = output_dir / "download"
    existing_videos = list(dl_dir.glob("*.mp4")) if dl_dir.exists() else []

    if existing_videos:
        logger.info("Using existing download: %s", existing_videos[0].name)
        import json

        audio_files = list(dl_dir.glob("*_audio.wav"))
        if not audio_files:
            raise FileNotFoundError("Downloaded video exists but audio WAV is missing.")
        frames_dir = dl_dir / "frames"
        frame_paths = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []

        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", str(existing_videos[0])],
                capture_output=True, text=True, check=True,
            )
            duration = float(json.loads(result.stdout)["format"]["duration"])
        except Exception:
            duration = 0.0

        return {
            "title": existing_videos[0].stem,
            "audio_path": audio_files[0],
            "frame_paths": frame_paths,
            "duration": duration,
            "metadata": {"url": url, "duration": duration},
        }

    dl_cfg = config.get("downloader", {})
    result = download_video(
        url=url,
        output_dir=str(dl_dir),
        frame_interval=dl_cfg.get("frame_interval", 15),
        max_frames=dl_cfg.get("max_frames", 120),
        video_format=dl_cfg.get("video_format",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"),
    )
    result["duration"] = result.get("metadata", {}).get("duration", 0)
    return result


def _transcribe(context: dict, config: dict) -> dict:
    """Run Whisper transcription and return transcript dict."""
    from tools.transcriber import transcribe_audio

    tc_cfg = config.get("transcriber", {})
    transcript = transcribe_audio(
        audio_path=context["audio_path"],
        model_name=tc_cfg.get("model_name", "tiny"),
        language=tc_cfg.get("language"),
        device=tc_cfg.get("device", "cpu"),
    )
    return transcript


# ── Public API ────────────────────────────────────────────────────────

def video_transcribe(
    url: str,
    config_path: str | Path = "config.yaml",
    output_dir: str | Path = "output",
) -> Path:
    """Download and transcribe a video, export to plain text."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    config = _load_config(config_path)
    out = Path(config.get("project", {}).get("output_dir", str(output_dir)))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        overall = progress.add_task("Transcription pipeline", total=4)

        # Step 1  Download
        progress.update(overall, description="[1/4] Downloading video …")
        context = _ensure_downloaded(url, config, out)
        progress.advance(overall)

        # Step 2  Whisper transcription
        progress.update(overall, description="[2/4] Transcribing audio (Whisper) …")
        transcript = _transcribe(context, config)
        progress.advance(overall)

        # Step 3  LLM refinement
        refine_cfg = config.get("transcript_refiner", {})
        if refine_cfg.get("enabled", True):
            progress.update(overall, description="[3/4] Refining transcript (LLM) …")
            from tools.transcript_refiner import refine_transcript
            transcript = refine_transcript(
                transcript,
                model=refine_cfg.get("model", "gemma2:27b"),
                host=refine_cfg.get("ollama_host", "http://localhost:11434"),
                progress=progress,
            )
        progress.advance(overall)

        # Step 4  Export text
        progress.update(overall, description="[4/4] Exporting transcript …")
        from tools.transcript_exporter import export_transcript_text

        txt_path = export_transcript_text(
            transcript=transcript,
            title=context.get("title", "Video Transcript"),
            metadata=context.get("metadata", {}),
            output_dir=str(out),
        )
        progress.advance(overall)

    logger.info("Transcript exported → %s", txt_path)
    return txt_path


def video_animate(
    url: str,
    config_path: str | Path = "config.yaml",
    output_dir: str | Path = "output",
) -> Path:
    """Run the full documentary animation pipeline."""
    config = _load_config(config_path)
    out = Path(config.get("project", {}).get("output_dir", str(output_dir)))

    # Import and run the documentary pipeline
    from run_documentary import reconstruct_context, run_pipeline, OUTPUT_DIR, DOWNLOAD_DIR

    context = _ensure_downloaded(url, config, out)

    # Transcribe
    transcript = _transcribe(context, config)
    context["full_text"] = transcript["full_text"]

    return run_pipeline(context, config)


# ── CLI ───────────────────────────────────────────────────────────────

@click.command("zuviscribe")
@click.argument("url")
@click.option("--animate", is_flag=True, help="Run the documentary animation pipeline.")
@click.option("--transcribe", is_flag=True, help="Export transcript to text file.")
@click.option("--config", "config_path", default="config.yaml", help="Path to config.yaml.")
@click.option("--output", "output_dir", default="output", help="Output directory.")
@click.option("--upload", is_flag=True, help="Upload the final video to YouTube.")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def cli(
    url: str,
    animate: bool,
    transcribe: bool,
    config_path: str,
    output_dir: str,
    upload: bool,
    verbose: bool,
) -> None:
    """Turn a YouTube video into a creative animated rendition.

    URL is the YouTube video link to process.

    \b
    Examples:
      python main.py URL --animate        # Create animated documentary
      python main.py URL --transcribe     # Export text transcript
      python main.py URL --animate --transcribe  # Both
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default to --animate if neither flag is given
    if not animate and not transcribe:
        logger.info("No mode specified. Use --animate or --transcribe. Defaulting to --animate.")
        animate = True

    logger.info("ZuviScribe starting …")
    logger.info("URL:        %s", url)
    logger.info("Mode:       %s", " + ".join(
        [m for m, f in [("animate", animate), ("transcribe", transcribe)] if f]))
    logger.info("Config:     %s", config_path)

    try:
        if transcribe:
            result_path = video_transcribe(
                url=url, config_path=config_path, output_dir=output_dir,
            )
            logger.info("✅ Transcript → %s", result_path)

        if animate:
            video_path = video_animate(
                url=url, config_path=config_path, output_dir=output_dir,
            )
            logger.info("✅ Documentary video → %s", video_path)

    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    cli()
