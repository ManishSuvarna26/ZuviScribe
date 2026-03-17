"""
agents.orchestrator
~~~~~~~~~~~~~~~~~~~
Async orchestrator that drives the full video-animation pipeline.

The orchestrator:

1. Loads ``config.yaml`` and merges CLI overrides.
2. Builds the pipeline from :mod:`agents.pipeline`.
3. Executes each step sequentially, passing a shared *context* dict.
4. In ``--dry-run`` mode, logs the planned actions without executing them.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml

from agents.pipeline import PipelineStep, build_pipeline

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read and return the YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file %s not found – using defaults.", config_path)
        return {}
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


class Orchestrator:
    """Coordinates the video-animation pipeline end-to-end."""

    def __init__(
        self,
        url: str,
        config: dict[str, Any] | None = None,
        dry_run: bool = False,
        upload: bool = False,
        output_dir: str | Path = "output",
    ) -> None:
        self.url = url
        self.config = config or load_config()
        self.dry_run = dry_run
        self.upload = upload
        self.output_dir = Path(output_dir)
        self.context: dict[str, Any] = {"url": url}

    # ── public API ────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Execute the full pipeline and return the final context dict."""
        pipeline = build_pipeline()

        # Skip upload step unless explicitly requested
        if not self.upload:
            pipeline = [s for s in pipeline if s.name != "upload"]

        for step in pipeline:
            if self.dry_run:
                self._log_dry_run(step)
                continue
            await self._execute_step(step)

        return self.context

    # ── internals ─────────────────────────────────────────────────────

    async def _execute_step(self, step: PipelineStep) -> None:
        """Run a single pipeline step, mapping context keys to function args."""
        logger.info("▶ %s – %s", step.name, step.description)
        kwargs = self._build_kwargs(step)

        # All tool functions are sync today; run in executor to keep the
        # event loop responsive and allow future async tools.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: step.fn(**kwargs))

        self._merge_result(step, result)
        logger.info("✔ %s complete.", step.name)

    def _build_kwargs(self, step: PipelineStep) -> dict[str, Any]:
        """Map the shared context + config into keyword arguments for *step.fn*."""
        cfg_key = step.config_key or step.name
        cfg_section = self.config.get(cfg_key, {})
        kwargs: dict[str, Any] = {}

        # Pull required keys from context
        for key in step.required_keys:
            if key not in self.context:
                raise RuntimeError(
                    f"Step '{step.name}' requires '{key}' but it is missing from the context."
                )
            kwargs[key] = self.context[key]

        # Inject common params
        if "output_dir" in _get_param_names(step.fn):
            sub_dir = self.output_dir / step.name
            kwargs["output_dir"] = str(sub_dir)

        # ── Step-specific wiring ──────────────────────────────────────
        # Forward original audio to synthesizer
        if step.name == "audio_synth" and "audio_path" in self.context:
            kwargs["original_audio"] = self.context["audio_path"]

        # Forward original video duration to storyboard generator
        if step.name == "storyboard":
            metadata = self.context.get("metadata", {})
            dur = metadata.get("duration")
            if dur:
                kwargs["video_duration"] = float(dur)

        # Construct output_path for the assembler
        if step.name == "assemble" and "output_path" not in kwargs:
            kwargs["output_path"] = str(self.output_dir / "final_output.mp4")

        # Merge config-section values (don't overwrite already-set keys)
        for k, v in cfg_section.items():
            if k not in kwargs:
                kwargs[k] = v

        return kwargs

    def _merge_result(self, step: PipelineStep, result: Any) -> None:
        """Store the step's output in the shared context."""
        if result is None:
            return

        if isinstance(result, dict):
            self.context.update(result)
        elif isinstance(result, list):
            # If the step declares exactly one produced key, use that name;
            # otherwise default to "scenes".
            if len(step.produced_keys) == 1:
                self.context[step.produced_keys[0]] = result
            else:
                self.context["scenes"] = result
        elif isinstance(result, Path):
            self.context["output_path"] = result
        else:
            self.context[step.name + "_result"] = result

    def _log_dry_run(self, step: PipelineStep) -> None:
        logger.info(
            "[DRY-RUN] Would execute '%s': %s (needs: %s → produces: %s)",
            step.name,
            step.description,
            ", ".join(step.required_keys) or "–",
            ", ".join(step.produced_keys) or "(mutates scenes)",
        )


def _get_param_names(fn: Any) -> set[str]:
    """Return the set of parameter names for *fn*."""
    import inspect
    try:
        sig = inspect.signature(fn)
        return set(sig.parameters)
    except (ValueError, TypeError):
        return set()
