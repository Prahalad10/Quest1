"""On-disk cache layout -- the single source of truth for paths.

    outputs/
        _resolve/{sha256(url)[:24]}.json   yt-dlp resolve, keyed by URL
        {media_key}/
            audio.wav          16kHz mono PCM, the ASR input
            audio.meta.json    which audio track that wav came from
            probe.json         ffprobe of the video stream + source_url
            transcript.json    words, timings, char offsets
            index.json         manifest: INDEX_VERSION, model, counts
            frames/frame_000004500.png

Per-video artifacts key on media_key (extractor + video id, stable). The
resolve cache cannot -- media_key is only known after resolving -- so it keys
on a hash of the URL.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src import config
from src.errors import InvalidInputError

_MEDIA_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{3,128}$")


def validate_media_key(media_key: str) -> str:
    """Reject anything that could escape OUTPUT_DIR.

    Not paranoia: api.py takes a media_key from an HTTP request, so without
    this a value like "../../etc/passwd" would reach the filesystem. Every
    path function below calls it, so they are safe by construction.
    """
    if not isinstance(media_key, str) or not media_key.strip():
        raise InvalidInputError("media_key is required and must be a non-empty string.")
    media_key = media_key.strip()
    if not _MEDIA_KEY_RE.match(media_key) or media_key in (".", ".."):
        raise InvalidInputError(
            f"Invalid media_key {media_key!r}: expected 3-128 chars of [A-Za-z0-9._-]."
        )
    return media_key


def media_dir(media_key: str) -> Path:
    return config.OUTPUT_DIR / validate_media_key(media_key)


def ensure_media_dir(media_key: str) -> Path:
    path = media_dir(media_key)
    path.mkdir(parents=True, exist_ok=True)
    return path


def audio_path(media_key: str) -> Path:
    return media_dir(media_key) / "audio.wav"


def audio_meta_path(media_key: str) -> Path:
    """Which track audio.wav came from; see audio.py:_audio_cache_is_stale."""
    return media_dir(media_key) / "audio.meta.json"


def probe_path(media_key: str) -> Path:
    return media_dir(media_key) / "probe.json"


def transcript_path(media_key: str) -> Path:
    return media_dir(media_key) / "transcript.json"


def index_path(media_key: str) -> Path:
    return media_dir(media_key) / "index.json"


def frames_dir(media_key: str) -> Path:
    return media_dir(media_key) / "frames"


def ensure_frames_dir(media_key: str) -> Path:
    path = frames_dir(media_key)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_cache_path(url: str) -> Path:
    """Hashed because URLs contain characters illegal in filenames."""
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:24]
    return config.OUTPUT_DIR / "_resolve" / f"{digest}.json"


def ensure_resolve_cache_dir() -> Path:
    path = config.OUTPUT_DIR / "_resolve"
    path.mkdir(parents=True, exist_ok=True)
    return path
