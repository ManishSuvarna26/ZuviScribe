"""Unit tests for tools.audio_synth."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.audio_synth import synthesize_audio


class TestSynthesizeAudio:
    @patch("tools.audio_synth._concat_audio")
    @patch("tools.audio_synth.subprocess.run")
    def test_split_original_mode(self, mock_run, mock_concat, sample_scenes, sample_audio, tmp_path):
        """original engine splits the source audio into segments."""
        results = synthesize_audio(
            scenes=sample_scenes,
            output_dir=tmp_path / "audio",
            engine="original",
            original_audio=sample_audio,
        )

        assert len(results) == len(sample_scenes)
        # One ffmpeg call per scene (split) 
        assert mock_run.call_count == len(sample_scenes)
        for r in results:
            assert "narration_audio_path" in r

    @patch("tools.audio_synth._concat_audio")
    @patch("tools.audio_synth._generate_silence")
    @patch("tools.audio_synth.TTS")
    def test_coqui_mode(self, mock_tts_cls, mock_silence, mock_concat, sample_scenes, tmp_path):
        """coqui engine generates speech from narration text."""
        mock_tts = MagicMock()
        mock_tts_cls.return_value = mock_tts

        results = synthesize_audio(
            scenes=sample_scenes,
            output_dir=tmp_path / "audio",
            engine="coqui",
        )

        assert len(results) == len(sample_scenes)
        # tts_to_file called for each scene with narration text
        assert mock_tts.tts_to_file.call_count == len(sample_scenes)
