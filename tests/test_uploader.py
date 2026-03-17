"""Unit tests for tools.uploader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.uploader import upload_to_youtube


class TestUploader:
    def test_raises_on_missing_video(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Video not found"):
            upload_to_youtube(tmp_path / "nope.mp4")

    def test_raises_on_missing_credentials(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00" * 10)

        with pytest.raises(FileNotFoundError, match="credentials"):
            upload_to_youtube(video, credentials_file=tmp_path / "missing.json")

    @patch("tools.uploader.MediaFileUpload")
    @patch("tools.uploader.build")
    @patch("tools.uploader.InstalledAppFlow")
    def test_successful_upload(self, mock_flow_cls, mock_build, mock_media, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00" * 10)
        creds = tmp_path / "client_secrets.json"
        creds.write_text("{}")

        # Mock OAuth flow
        mock_flow = MagicMock()
        mock_flow_cls.from_client_secrets_file.return_value = mock_flow
        mock_flow.run_local_server.return_value = MagicMock()

        # Mock YouTube API
        mock_yt = MagicMock()
        mock_build.return_value = mock_yt
        mock_insert = MagicMock()
        mock_yt.videos.return_value.insert.return_value = mock_insert
        mock_insert.next_chunk.return_value = (None, {"id": "abc123"})

        result = upload_to_youtube(video, credentials_file=creds)

        assert result["video_id"] == "abc123"
        assert "youtube.com" in result["url"]
