"""
tools.uploader
~~~~~~~~~~~~~~
Upload the final video to YouTube (or other platforms) via the YouTube Data API v3.

Inputs
------
- video_path    : str | Path – Path to the final .mp4 file.
- title         : str        – Video title.
- description   : str        – Video description.
- tags          : list[str]  – Tags / keywords.
- category_id   : str        – YouTube category ID (default "22" – People & Blogs).
- privacy       : str        – ``"private"``, ``"unlisted"``, or ``"public"``.
- credentials   : str | Path – Path to ``client_secrets.json``.

Outputs
-------
dict with keys:
    video_id  : str  – YouTube video ID of the uploaded video.
    url       : str  – Direct URL to the video.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
_YOUTUBE_API_SERVICE_NAME = "youtube"
_YOUTUBE_API_VERSION = "v3"


def upload_to_youtube(
    video_path: str | Path,
    title: str = "Animated Video",
    description: str = "",
    tags: list[str] | None = None,
    category_id: str = "22",
    privacy: str = "private",
    credentials_file: str | Path = "client_secrets.json",
) -> dict[str, str]:
    """Authenticate with YouTube and upload *video_path*.

    Uses OAuth 2.0 installed-app flow.  On first run a browser window will open
    for consent; the resulting token is cached locally.

    Returns a dict with ``video_id`` and ``url``.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    video_path = Path(video_path)
    credentials_file = Path(credentials_file)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not credentials_file.exists():
        raise FileNotFoundError(
            f"OAuth credentials not found: {credentials_file}. "
            "Download client_secrets.json from the Google Cloud Console."
        )

    logger.info("Authenticating with YouTube …")
    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_file), scopes=[_YOUTUBE_UPLOAD_SCOPE]
    )
    creds = flow.run_local_server(port=0)

    youtube = build(
        _YOUTUBE_API_SERVICE_NAME,
        _YOUTUBE_API_VERSION,
        credentials=creds,
    )

    body: dict[str, Any] = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)

    logger.info("Uploading %s to YouTube …", video_path.name)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        _, response = request.next_chunk()

    video_id: str = response["id"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    logger.info("Upload complete → %s", url)

    return {"video_id": video_id, "url": url}
