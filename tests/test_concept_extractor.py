"""Unit tests for tools.concept_extractor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.concept_extractor import _parse_response, extract_concepts


class TestParseResponse:
    def test_extracts_description_and_concepts(self):
        text = (
            "A man walks through a sunny desert.\n"
            "Concepts: desert, man, sun, walking, sand"
        )
        desc, concepts = _parse_response(text)
        assert "desert" in desc
        assert concepts == ["desert", "man", "sun", "walking", "sand"]

    def test_no_concepts_line(self):
        desc, concepts = _parse_response("Just a description.")
        assert desc == "Just a description."
        assert concepts == []


class TestExtractConcepts:
    @patch("tools.concept_extractor._ollama")
    def test_processes_all_frames(self, mock_ollama_mod, sample_frames):
        mock_client = MagicMock()
        mock_ollama_mod.Client.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {
                "content": (
                    "A bright sunny frame.\nConcepts: sun, sky, grass"
                )
            }
        }

        results = extract_concepts(sample_frames, model="llava", ollama_host="http://fake:11434")

        assert len(results) == len(sample_frames)
        assert results[0]["concepts"] == ["sun", "sky", "grass"]
        assert mock_client.chat.call_count == len(sample_frames)
