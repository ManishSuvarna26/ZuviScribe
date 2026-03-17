"""Integration test – dry-run pipeline (no external services)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agents.orchestrator import Orchestrator


class TestDryRunPipeline:
    def test_dry_run_completes_without_error(self):
        """--dry-run should log every step without calling any tool."""
        orch = Orchestrator(
            url="https://www.youtube.com/watch?v=FAKE",
            config={},
            dry_run=True,
            upload=False,
            output_dir="/tmp/test_output",
        )
        ctx = asyncio.run(orch.run())

        # Context should still have the URL and nothing else
        assert ctx["url"] == "https://www.youtube.com/watch?v=FAKE"
        # No output_path because nothing was actually executed
        assert "output_path" not in ctx

    def test_dry_run_with_upload_flag(self):
        orch = Orchestrator(
            url="https://www.youtube.com/watch?v=FAKE",
            config={},
            dry_run=True,
            upload=True,
            output_dir="/tmp/test_output",
        )
        ctx = asyncio.run(orch.run())
        assert ctx["url"] == "https://www.youtube.com/watch?v=FAKE"
