"""Unit tests for tools.storyboard_generator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.storyboard_generator import _parse_storyboard, generate_storyboard


class TestParseStoryboard:
    def test_valid_json(self):
        raw = json.dumps([
            {"scene_number": 1, "description": "d1", "narration": "n1", "duration_hint": 10},
            {"scene_number": 2, "description": "d2", "narration": "n2", "duration_hint": 8},
        ])
        scenes = _parse_storyboard(raw, min_scenes=1, max_scenes=10)
        assert len(scenes) == 2
        assert scenes[0]["scene_number"] == 1

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps([{"description": "x"}]) + "\n```"
        scenes = _parse_storyboard(raw, min_scenes=1, max_scenes=10)
        assert len(scenes) == 1

    def test_raises_on_garbage(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_storyboard("not json at all", min_scenes=1, max_scenes=10)

    def test_caps_at_max_scenes(self):
        raw = json.dumps([{"description": f"s{i}"} for i in range(20)])
        scenes = _parse_storyboard(raw, min_scenes=1, max_scenes=5)
        assert len(scenes) == 5


class TestGenerateStoryboard:
    @patch("tools.storyboard_generator._ollama")
    def test_returns_scene_list(self, mock_mod, sample_concepts):
        mock_client = MagicMock()
        mock_mod.Client.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {
                "content": json.dumps([
                    {
                        "scene_number": 1,
                        "description": "Cyber desert",
                        "narration": "In a distant land…",
                        "duration_hint": 12,
                    },
                ])
            }
        }

        scenes = generate_storyboard(
            transcript="Hello world.",
            concepts=sample_concepts,
            model="llama3",
            ollama_host="http://fake:11434",
        )

        assert len(scenes) == 1
        assert scenes[0]["description"] == "Cyber desert"
