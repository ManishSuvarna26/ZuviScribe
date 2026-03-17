#!/usr/bin/env python3
"""
run_documentary.py – Generate a documentary-style video from a video URL.

Pipeline:

  Transcribe → LLM Documentary Script → TTS Narration (drives timing)
    → SDXL Image Generation → Cinematic Animation → Professional Assembly

The result is a narration-driven documentary in the style of Fern, LEMMiNO,
or Kurzgesagt  rich narration over cinematic visuals with camera movements,
color grading, and smooth transitions.
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
logger = logging.getLogger("documentary")

OUTPUT_DIR = Path("output")
DOWNLOAD_DIR = OUTPUT_DIR / "download"


def _save_storyboard(scenes: list, path: Path) -> None:
    """Write a human-readable storyboard file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 72,
        "DOCUMENTARY STORYBOARD",
        "=" * 72,
        "",
    ]
    for s in scenes:
        num = s.get("scene_number", "?")
        mood = s.get("mood", "?")
        narration = s.get("narration", "")
        visual = s.get("visual_prompt", "")
        lines.append(f"── Scene {num}  [{mood.upper()}] " + "─" * 40)
        lines.append("")
        lines.append("NARRATION:")
        lines.append(narration)
        lines.append("")
        lines.append("VISUAL PROMPT:")
        lines.append(visual)
        lines.append("")
        lines.append("")
    lines.append("=" * 72)
    lines.append(f"Total scenes: {len(scenes)}")
    lines.append("=" * 72)
    path.write_text("\n".join(lines))


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f) or {}


def _get_video_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def reconstruct_context() -> dict:
    """Rebuild context from files on disk (video, audio, frames)."""
    videos = list(DOWNLOAD_DIR.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError("No .mp4 found in output/download/")
    video = videos[0]

    audio_files = list(DOWNLOAD_DIR.glob("*_audio.wav"))
    if not audio_files:
        raise FileNotFoundError("No audio WAV found in output/download/")
    audio_path = audio_files[0]

    frames_dir = DOWNLOAD_DIR / "frames"
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    duration = _get_video_duration(video)

    logger.info("Video: %s (%.1fs), Frames: %d", video.name, duration, len(frame_paths))

    # Re-run Whisper transcription
    logger.info("Transcribing audio with Whisper…")
    from tools.transcriber import transcribe_audio
    transcript = transcribe_audio(audio_path=audio_path, model_name="tiny", device="cpu")

    return {
        "title": video.stem,
        "audio_path": audio_path,
        "frame_paths": frame_paths,
        "duration": duration,
        "full_text": transcript["full_text"],
    }


def run_pipeline(context: dict, config: dict) -> Path:
    """Execute the full documentary pipeline."""

    # ── Step 1: Concept Extraction (quick, for context) ──────────────
    logger.info("═══ Step 1/8: Extracting visual concepts ═══")
    from tools.concept_extractor import extract_concepts

    ce_cfg = config.get("concept_extractor", {})
    concepts = extract_concepts(
        frame_paths=context["frame_paths"],
        model=ce_cfg.get("model", "llava:7b"),
        ollama_host=ce_cfg.get("ollama_host", "http://localhost:11434"),
        max_concepts_per_frame=ce_cfg.get("max_concepts_per_frame", 5),
        max_frames_to_analyse=ce_cfg.get("max_frames_to_analyse", 15),
    )
    logger.info("Extracted concepts from %d frames.", len(concepts))

    # ── Step 2: Documentary Script ────────────────────────────────────
    logger.info("═══ Step 2/8: Writing documentary script ═══")
    from tools.script_writer import write_script

    sb_cfg = config.get("storyboard", {})
    scenes = write_script(
        full_text=context["full_text"],
        concepts=concepts,
        model=sb_cfg.get("model", "gemma2:27b"),
        ollama_host=sb_cfg.get("ollama_host", "http://localhost:11434"),
        min_scenes=sb_cfg.get("min_scenes", 6),
        max_scenes=sb_cfg.get("max_scenes", 10),
    )

    for s in scenes:
        logger.info("  Scene %d [%s]: %s", s["scene_number"], s["mood"],
                     s["narration"][:80] + "…" if len(s["narration"]) > 80 else s["narration"])

    # Save storyboard to a readable text file
    storyboard_path = OUTPUT_DIR / "storyboard.txt"
    _save_storyboard(scenes, storyboard_path)
    logger.info("Storyboard saved → %s", storyboard_path)

    # Also save raw JSON for machine consumption
    storyboard_json = OUTPUT_DIR / "storyboard.json"
    with open(storyboard_json, "w") as f:
        json.dump(scenes, f, indent=2, default=str)

    # ── Step 3: TTS Narration ─────────────────────────────────────────
    logger.info("═══ Step 3/8: Generating TTS narration ═══")
    from tools.narration_tts import generate_narration

    tts_cfg = config.get("narration", {})
    scenes = generate_narration(
        scenes=scenes,
        output_dir=str(OUTPUT_DIR / "narration"),
        voice=tts_cfg.get("voice", "en-US-ChristopherNeural"),
    )

    total_dur = sum(s["narration_duration"] for s in scenes)
    logger.info("Total narration duration: %.1fs across %d scenes.", total_dur, len(scenes))

    # ── Step 4: SDXL Image Rendering (3 variants per scene) ───────────
    logger.info("═══ Step 4/8: Rendering images with SDXL (3 variants/scene) ═══")
    from tools.image_renderer import render_images, cleanup_pipeline

    ir_cfg = config.get("image_renderer", {})
    scenes = render_images(
        scenes=scenes,
        output_dir=str(OUTPUT_DIR / "render_images"),
        model=ir_cfg.get("model", "stabilityai/sdxl-turbo"),
        width=ir_cfg.get("width", 1024),
        height=ir_cfg.get("height", 576),
        steps=ir_cfg.get("steps", 6),
        cfg_scale=ir_cfg.get("cfg_scale", 0.0),
        negative_prompt=ir_cfg.get("negative_prompt",
            "blurry, low quality, watermark, text, words, letters, ugly, deformed, amateur"),
        variants_per_scene=ir_cfg.get("variants_per_scene", 3),
    )

    # Free SDXL GPU memory before loading MusicGen
    cleanup_pipeline()
    logger.info("SDXL pipeline freed for next step.")

    # ── Step 5: Background Music Generation ───────────────────────────
    music_path = None
    music_cfg = config.get("music", {})
    if music_cfg.get("enabled", True):
        logger.info("═══ Step 5/8: Generating background music ═══")
        from tools.music_generator import generate_background_music

        # Pick dominant mood from scenes for music style
        mood_counts: dict[str, int] = {}
        for s in scenes:
            m = s.get("mood", "contemplative")
            mood_counts[m] = mood_counts.get(m, 0) + 1
        dominant_mood = max(mood_counts, key=mood_counts.get)
        logger.info("Dominant mood: %s", dominant_mood)

        try:
            music_path = generate_background_music(
                total_duration=total_dur + 5,
                output_path=str(OUTPUT_DIR / "background_music.wav"),
                mood=dominant_mood,
            )
        except Exception as exc:
            logger.warning("Music generation failed: %s. Continuing without music.", exc)
            music_path = None
    else:
        logger.info("═══ Step 5/8: Background music disabled, skipping ═══")

    # ── Step 6: Cinematic Animation (SVD i2v + FFmpeg) ──────────────
    logger.info("═══ Step 6/8: Creating cinematic clips (SVD video gen) ═══")
    from tools.wan_animator import animate_scenes

    anim_cfg = config.get("animation", {})
    scenes = animate_scenes(
        scenes=scenes,
        output_dir=str(OUTPUT_DIR / "animate"),
        fps=anim_cfg.get("fps", 24),
        width=1920,
        height=1080,
    )

    # ── Step 7: Final Assembly ────────────────────────────────────────
    logger.info("═══ Step 7/8: Assembling final documentary ═══")
    from tools.documentary_assembler import assemble_documentary

    va_cfg = config.get("video_assembly", {})
    output_video = assemble_documentary(
        scenes=scenes,
        output_path=str(OUTPUT_DIR / "documentary_output.mp4"),
        transition=va_cfg.get("transition", "crossfade"),
        transition_dur=va_cfg.get("transition_dur", 0.8),
        codec=va_cfg.get("codec", "libx264"),
        crf=va_cfg.get("crf", 18),
        music_path=str(music_path) if music_path else None,
        music_volume=music_cfg.get("volume", 0.12),
    )

    # ── Step 8: Summary ───────────────────────────────────────────────
    logger.info("═══ Step 8/8: Pipeline complete ═══")
    n_images = sum(len(s.get("image_paths", [s.get("image_path")])) for s in scenes)
    logger.info("  Scenes: %d  |  Images: %d  |  Duration: %.1fs  |  Music: %s",
                len(scenes), n_images, total_dur,
                "yes" if music_path else "no")

    return Path(output_video)


def main():
    config = load_config()
    context = reconstruct_context()
    output = run_pipeline(context, config)
    logger.info("🎬 Documentary video: %s", output)


if __name__ == "__main__":
    main()
