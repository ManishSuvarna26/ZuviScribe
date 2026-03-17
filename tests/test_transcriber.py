"""Unit tests for tools.transcriber."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.transcriber import transcribe_audio


class TestTranscribeAudio:
    @patch("tools.transcriber.whisper")
    def test_returns_structured_result(self, mock_whisper, sample_audio):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": " Hello world. This is a test. ",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " Hello world."},
                {"start": 2.0, "end": 4.0, "text": " This is a test."},
            ],
            "language": "en",
        }
        mock_whisper.load_model.return_value = mock_model

        result = transcribe_audio(sample_audio, model_name="base", device="cpu")

        assert result["full_text"] == "Hello world. This is a test."
        assert len(result["segments"]) == 2
        assert result["segments"][0]["start"] == 0.0
        assert result["language"] == "en"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            transcribe_audio(tmp_path / "nonexistent.wav")
