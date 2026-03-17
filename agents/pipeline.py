"""
agents.pipeline
~~~~~~~~~~~~~~~
Defines the sequential pipeline of tool invocations that drive the
video-animation workflow.  Each *step* wraps one tool module and carries
metadata (name, callable, required inputs, produced outputs) so the
orchestrator can validate data flow and implement ``--dry-run``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable


@dataclasses.dataclass(frozen=True)
class PipelineStep:
    """One stage in the video-animation pipeline."""

    name: str
    description: str
    fn: Callable[..., Any]
    required_keys: tuple[str, ...]  # keys that must exist in the context dict
    produced_keys: tuple[str, ...]  # keys the step adds to the context dict
    config_key: str = ""  # config.yaml section name; defaults to step name if empty


def build_pipeline() -> list[PipelineStep]:
    """Return the ordered default pipeline steps.

    Each step is lazily imported so that heavy dependencies (torch, whisper …)
    are only loaded when the step actually runs.
    """
    from tools.downloader import download_video
    from tools.transcriber import transcribe_audio
    from tools.concept_extractor import extract_concepts
    from tools.storyboard_generator import generate_storyboard
    from tools.image_renderer import render_images
    from tools.animation_engine import animate_scenes
    from tools.audio_synth import synthesize_audio
    from tools.video_assembler import assemble_video
    from tools.uploader import upload_to_youtube

    return [
        PipelineStep(
            name="download",
            description="Download YouTube video, extract audio & frames.",
            fn=download_video,
            required_keys=("url",),
            produced_keys=("title", "audio_path", "frame_paths", "metadata"),
            config_key="downloader",
        ),
        PipelineStep(
            name="transcribe",
            description="Transcribe audio to text with Whisper.",
            fn=transcribe_audio,
            required_keys=("audio_path",),
            produced_keys=("full_text", "segments", "language"),
            config_key="transcriber",
        ),
        PipelineStep(
            name="extract_concepts",
            description="Extract visual concepts from video frames.",
            fn=extract_concepts,
            required_keys=("frame_paths",),
            produced_keys=("concepts",),
            config_key="concept_extractor",
        ),
        PipelineStep(
            name="storyboard",
            description="Generate a creative animated storyboard.",
            fn=generate_storyboard,
            required_keys=("full_text", "concepts"),
            produced_keys=("scenes",),
        ),
        PipelineStep(
            name="render_images",
            description="Render images for each storyboard scene via SDXL.",
            fn=render_images,
            required_keys=("scenes",),
            produced_keys=(),  # mutates scenes in-place (adds image_path)
            config_key="image_renderer",
        ),
        PipelineStep(
            name="animate",
            description="Turn static images into animated clips.",
            fn=animate_scenes,
            required_keys=("scenes",),
            produced_keys=(),  # mutates scenes (adds clip_path)
            config_key="animation",
        ),
        PipelineStep(
            name="audio_synth",
            description="Synthesize narration audio for each scene.",
            fn=synthesize_audio,
            required_keys=("scenes", "audio_path"),  # audio_path forwarded for original-mode
            produced_keys=(),  # mutates scenes (adds narration_audio_path)
        ),
        PipelineStep(
            name="assemble",
            description="Assemble clips + narration into final video.",
            fn=assemble_video,
            required_keys=("scenes",),
            produced_keys=("output_path",),
            config_key="video_assembly",
        ),
        PipelineStep(
            name="upload",
            description="Upload the final video to YouTube.",
            fn=upload_to_youtube,
            required_keys=("output_path",),
            produced_keys=("video_id", "video_url"),
            config_key="uploader",
        ),
    ]
