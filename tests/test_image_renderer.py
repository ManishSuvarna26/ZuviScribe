"""Unit tests for tools.image_renderer."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.image_renderer import render_images


class TestRenderImages:
    @patch("tools.image_renderer._ollama")
    def test_saves_images(self, mock_mod, sample_scenes, tmp_path):
        mock_client = MagicMock()
        mock_mod.Client.return_value = mock_client

        # Fake 1x1 white PNG encoded as base64
        fake_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100).decode()
        mock_client.generate.return_value = {"images": [fake_png]}

        results = render_images(
            scenes=sample_scenes,
            output_dir=tmp_path / "images",
            model="sdxl",
            ollama_host="http://fake:11434",
        )

        assert len(results) == len(sample_scenes)
        for r in results:
            assert "image_path" in r
            assert Path(r["image_path"]).exists()
