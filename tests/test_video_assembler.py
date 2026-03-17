"""Unit tests for tools.video_assembler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.video_assembler import assemble_video


class TestAssembleVideo:
    @patch("tools.video_assembler.subprocess.run")
    def test_ffmpeg_concat(self, mock_run, sample_scenes, tmp_path):
        """FFmpeg-only assembly path should call subprocess multiple times."""
        # Add fake clip + narration paths
        for s in sample_scenes:
            clip = tmp_path / f"clip_{s['scene_number']}.mp4"
            narr = tmp_path / f"narr_{s['scene_number']}.wav"
            clip.write_bytes(b"\x00" * 10)
            narr.write_bytes(b"\x00" * 10)
            s["clip_path"] = clip
            s["narration_audio_path"] = narr

        out = tmp_path / "final.mp4"
        result = assemble_video(scenes=sample_scenes, output_path=out, transition="cut")

        assert result == out
        # At least: concat clips + concat narration + mux
        assert mock_run.call_count >= 3
