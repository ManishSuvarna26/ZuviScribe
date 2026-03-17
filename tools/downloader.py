"""
tools.downloader
~~~~~~~~~~~~~~~~
Download a YouTube video, extract audio and key frames.

Inputs
------
- url : str – YouTube video URL.
- output_dir : str – Directory to store downloaded assets.
- frame_interval : int – Extract one frame every *N* seconds (default 2).
- max_frames : int – Maximum number of frames to extract (default 120).

Outputs
-------
dict with keys:
    title       : str           – Video title (sanitised for filenames).
    audio_path  : pathlib.Path  – Path to the extracted audio (.wav).
    frame_paths : list[Path]    – Ordered list of extracted JPEG frame paths.
    metadata    : dict          – yt-dlp info dict (duration, description …).
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _sanitise_filename(name: str) -> str:
    """Remove characters that are unsafe for file-system paths."""
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')[:120]


def download_video(
    url: str,
    output_dir: str | Path = "output",
    frame_interval: int = 2,
    max_frames: int = 120,
    video_format: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
) -> dict[str, Any]:
    """Download video from *url*, extract audio and evenly-spaced frames.

    The function shells out to **yt-dlp** for the download and **ffmpeg** for
    audio / frame extraction.  Both must be available on ``$PATH``.

    Returns a dict described in the module docstring.
    """
    import yt_dlp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download with yt-dlp ───────────────────────────────────────
    base_opts: dict[str, Any] = {
        "format": video_format,
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,  # download only the specific video, not the whole playlist
    }

    # Attempt order: ios client (best 403 bypass) → android → default (no UA tricks)
    attempt_configs = [
        {"extractor_args": {"youtube": {"player_client": ["ios"]}}},
        {"extractor_args": {"youtube": {"player_client": ["android"]}}},
        {},  # plain attempt as last resort
    ]

    info = None
    last_exc: Exception | None = None
    for extra in attempt_configs:
        try:
            opts = {**base_opts, **extra}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.DownloadError as exc:
            last_exc = exc
            logger.warning("Download attempt failed (%s), trying next strategy …", exc)
            continue

    if info is None:
        raise RuntimeError(
            f"yt-dlp could not download {url} after multiple attempts. "
            f"Last error: {last_exc}"
        )

    title_raw: str = info.get("title", "video")
    title = _sanitise_filename(title_raw)

    video_path = next(output_dir.glob("*.mp4"), None) or next(output_dir.glob("*.webm"), None)
    if video_path is None:
        raise FileNotFoundError("Downloaded video not found in output_dir")

    logger.info("Downloaded: %s → %s", title_raw, video_path)

    # ── 2. Extract audio (.wav) ───────────────────────────────────────
    audio_path = output_dir / f"{title}_audio.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )
    logger.info("Audio extracted → %s", audio_path)

    # ── 3. Extract frames ─────────────────────────────────────────────
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps=1/{frame_interval}",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            str(frames_dir / f"{title}_frame_%04d.jpg"),
        ],
        check=True,
        capture_output=True,
    )

    frame_paths = sorted(frames_dir.glob(f"{title}_frame_*.jpg"))
    logger.info("Extracted %d frames → %s", len(frame_paths), frames_dir)

    return {
        "title": title,
        "audio_path": audio_path,
        "frame_paths": frame_paths,
        "metadata": {
            "duration": info.get("duration"),
            "description": info.get("description", ""),
            "uploader": info.get("uploader", ""),
            "url": url,
        },
    }
