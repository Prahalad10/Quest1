"""On-disk layout for per-video artifacts.

Every path the pipeline writes goes through here, so the cache layout is
described in exactly one place:

    data/{media_key}/
        audio.wav        16kHz mono PCM, the ASR input
        probe.json       ffprobe of the VIDEO stream (fps, VFR flag, dimensions)
        transcript.json  raw ASR output + word timings
        index.json       manifest: INDEX_VERSION, normalized text, char offsets
        frames/          extracted PNGs, one per answered query
"""

from __future__ import annotations

import re
from pathlib import Path

from app import config
from app.errors import InvalidInputError

# media_key is built by resolve.compute_media_key; anything else is a caller bug
# and must never be turned into a filesystem path.
_MEDIA_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{3,128}$")


def validate_media_key(media_key: str) -> str:
    """Reject anything that could escape DATA_DIR or confuse the filesystem."""
    if not isinstance(media_key, str) or not media_key.strip():
        raise InvalidInputError("media_key is required and must be a non-empty string.")
    media_key = media_key.strip()
    if not _MEDIA_KEY_RE.match(media_key) or media_key in (".", ".."):
        raise InvalidInputError(
            f"Invalid media_key {media_key!r}: expected 3-128 chars of [A-Za-z0-9._-]."
        )
    return media_key


def media_dir(media_key: str) -> Path:
    return config.DATA_DIR / validate_media_key(media_key)


def ensure_media_dir(media_key: str) -> Path:
    path = media_dir(media_key)
    path.mkdir(parents=True, exist_ok=True)
    return path


def audio_path(media_key: str) -> Path:
    return media_dir(media_key) / "audio.wav"


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
