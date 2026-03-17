"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def tmp_output(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture()
def sample_frames(tmp_path: Path) -> list[Path]:
    """Create dummy JPEG frame files and return their paths."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    paths = []
    for i in range(3):
        p = frames_dir / f"frame_{i:04d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header
        paths.append(p)
    return paths


@pytest.fixture()
def sample_audio(tmp_path: Path) -> Path:
    """Create a dummy WAV file."""
    p = tmp_path / "audio.wav"
    # Minimal WAV header (44 bytes) + silence
    header = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt "
    header += (16).to_bytes(4, "little")  # chunk size
    header += (1).to_bytes(2, "little")   # PCM
    header += (1).to_bytes(2, "little")   # mono
    header += (16000).to_bytes(4, "little")  # sample rate
    header += (32000).to_bytes(4, "little")  # byte rate
    header += (2).to_bytes(2, "little")   # block align
    header += (16).to_bytes(2, "little")  # bits per sample
    header += b"data" + (0).to_bytes(4, "little")
    p.write_bytes(header)
    return p


@pytest.fixture()
def sample_scenes() -> list[dict]:
    """A minimal storyboard scene list for testing downstream tools."""
    return [
        {
            "scene_number": 1,
            "description": "A desert with a floating neon-lit city in cyberpunk style",
            "narration": "In the far future, cities float above endless dunes.",
            "duration_hint": 5.0,
        },
        {
            "scene_number": 2,
            "description": "A forest of bioluminescent trees with tiny robots",
            "narration": "Tiny robots tend the glowing forest.",
            "duration_hint": 5.0,
        },
    ]


@pytest.fixture()
def sample_concepts() -> list[dict]:
    """Concept extractor output for testing the storyboard generator."""
    return [
        {
            "frame_index": 0,
            "path": Path("/fake/frame_0000.jpg"),
            "description": "A man standing in a desert landscape.",
            "concepts": ["desert", "man", "sand", "sky"],
        },
        {
            "frame_index": 1,
            "path": Path("/fake/frame_0001.jpg"),
            "description": "A city skyline at night.",
            "concepts": ["city", "night", "lights", "buildings"],
        },
    ]
