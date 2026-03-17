"""Unit tests for tools.downloader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.downloader import _sanitise_filename, download_video


class TestSanitiseFilename:
    def test_removes_special_chars(self):
        assert _sanitise_filename("Hello, World! (2024)") == "Hello_World_2024"

    def test_truncates_long_names(self):
        long = "a" * 200
        assert len(_sanitise_filename(long)) <= 120

    def test_strips_whitespace(self):
        assert _sanitise_filename("  padded  ") == "padded"


class TestDownloadVideo:
    @patch("tools.downloader.subprocess.run")
    @patch("tools.downloader.yt_dlp.YoutubeDL")
    def test_happy_path(self, mock_ydl_cls, mock_run, tmp_path):
        # Arrange: yt-dlp returns info
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "title": "Test Video",
            "duration": 60,
            "description": "A test.",
            "uploader": "Tester",
        }
        mock_ydl_cls.return_value = mock_ydl

        # Create a fake mp4 so glob finds it
        fake_mp4 = tmp_path / "Test_Video.mp4"
        fake_mp4.write_bytes(b"\x00" * 10)

        # Also create expected audio output so assertions pass
        result = download_video(
            url="https://www.youtube.com/watch?v=TEST",
            output_dir=tmp_path,
            frame_interval=2,
            max_frames=5,
        )

        assert result["title"] == "Test_Video"
        assert mock_run.call_count == 2  # ffmpeg audio + frames
        assert result["metadata"]["duration"] == 60

    @patch("tools.downloader.yt_dlp.YoutubeDL")
    def test_raises_on_no_info(self, mock_ydl_cls, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = None
        mock_ydl_cls.return_value = mock_ydl

        with pytest.raises(RuntimeError, match="no info"):
            download_video("https://youtube.com/watch?v=BAD", output_dir=tmp_path)
