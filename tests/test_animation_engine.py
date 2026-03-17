"""Unit tests for tools.animation_engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.animation_engine import animate_scenes


class TestAnimateScenes:
    @patch("tools.animation_engine.subprocess.run")
    def test_ken_burns_fallback(self, mock_run, sample_scenes, tmp_path):
        """When animatediff-cli is not found, fall back to FFmpeg Ken Burns."""
        # Add fake image_path to scenes
        for s in sample_scenes:
            img = tmp_path / f"scene_{s['scene_number']}.png"
            img.write_bytes(b"\x89PNG" + b"\x00" * 50)
            s["image_path"] = img

        # First call (animatediff-cli) raises FileNotFoundError,
        # second call (ffmpeg) succeeds.
        def side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if cmd[0] == "animatediff-cli":
                raise FileNotFoundError("not installed")
            return None  # ffmpeg "succeeds"

        mock_run.side_effect = side_effect

        results = animate_scenes(
            scenes=sample_scenes,
            output_dir=tmp_path / "clips",
            method="animatediff",
            fps=24,
        )

        assert len(results) == len(sample_scenes)
        for r in results:
            assert "clip_path" in r

    @patch("tools.animation_engine.subprocess.run")
    def test_img2vid_method(self, mock_run, sample_scenes, tmp_path):
        for s in sample_scenes:
            img = tmp_path / f"scene_{s['scene_number']}.png"
            img.write_bytes(b"\x89PNG" + b"\x00" * 50)
            s["image_path"] = img

        results = animate_scenes(
            scenes=sample_scenes,
            output_dir=tmp_path / "clips",
            method="img2vid",
        )

        assert len(results) == len(sample_scenes)
        # FFmpeg should have been called once per scene
        assert mock_run.call_count == len(sample_scenes)
