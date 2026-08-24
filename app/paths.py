"""On-disk layout for cached artifacts -- the single source of truth for paths.

WHY THIS MODULE EXISTS
    Six modules read and write files under data/. If each built its own paths
    with string concatenation, changing the layout would mean hunting through
    all of them, and a typo would silently create a second cache directory that
    never gets a hit. Everything goes through these functions instead.

THE LAYOUT

    data/
        _resolve/
            {sha256(url)[:24]}.json   cached yt-dlp resolve, keyed by URL
        {media_key}/
            audio.wav                 16kHz mono PCM, the ASR input
            probe.json                ffprobe of the VIDEO stream + source_url
            transcript.json           ASR output, word timings, char offsets
            index.json                manifest: INDEX_VERSION, model, counts
            frames/
                frame_000004500.png   one PNG per answered timestamp

WHY TWO KEYING SCHEMES
    Per-video artifacts are keyed by `media_key` (extractor + video id), which
    is stable across runs. The resolve cache cannot use it -- the media_key is
    only KNOWN after resolving -- so it is keyed by a hash of the URL instead.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app import config
from app.errors import InvalidInputError

# media_key is built by resolve.compute_media_key; anything else is a caller bug
# and must never be turned into a filesystem path.
_MEDIA_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{3,128}$")


def validate_media_key(media_key: str) -> str:
    """Reject anything that could escape DATA_DIR or confuse the filesystem.

    WHY THIS IS NOT PARANOIA: in the web service (app/api.py) a media_key can
    reach the filesystem from an HTTP request. Without this check a value like
    "../../etc/passwd" would be interpolated straight into a path. Validating
    here means every path function below is safe by construction.

    USED BY: every path function in this module, plus index.py, matching.py and
    frame.py before they touch a media_key at all.
    """
    if not isinstance(media_key, str) or not media_key.strip():
        raise InvalidInputError("media_key is required and must be a non-empty string.")
    media_key = media_key.strip()
    if not _MEDIA_KEY_RE.match(media_key) or media_key in (".", ".."):
        raise InvalidInputError(
            f"Invalid media_key {media_key!r}: expected 3-128 chars of [A-Za-z0-9._-]."
        )
    return media_key


# --- Per-video artifact paths ------------------------------------------------
# All of these validate first, so passing a hostile media_key raises rather than
# returning a path that points outside data/.

def media_dir(media_key: str) -> Path:
    """Root directory for one video's cached artifacts.

    USED BY: every other path function here.
    """
    return config.DATA_DIR / validate_media_key(media_key)


def ensure_media_dir(media_key: str) -> Path:
    """media_dir(), creating it if absent.

    USED BY: audio.py and index.py before writing their first artifact.
    """
    path = media_dir(media_key)
    path.mkdir(parents=True, exist_ok=True)
    return path


def audio_path(media_key: str) -> Path:
    """The 16kHz mono wav that ASR reads.

    USED BY: audio.py (writes it), index.py (reads it), service.py (checks
    existence to report whether the audio stage was cached).
    """
    return media_dir(media_key) / "audio.wav"


def probe_path(media_key: str) -> Path:
    """ffprobe results for the video stream: fps, dimensions, VFR flag, source_url.

    USED BY: audio.py (writes it), frame.py (reads it for frame arithmetic),
    service.py (cache reporting).
    """
    return media_dir(media_key) / "probe.json"


def transcript_path(media_key: str) -> Path:
    """Full ASR output: segments, words, normalized text, char offsets.

    USED BY: index.py only. Everything else goes through TranscriptIndex.
    """
    return media_dir(media_key) / "transcript.json"


def index_path(media_key: str) -> Path:
    """The manifest whose INDEX_VERSION decides if the transcript is still valid.

    Kept separate from transcript.json so staleness can be checked by reading a
    few hundred bytes instead of parsing a possibly multi-megabyte transcript.

    USED BY: index.py (load_index, _persist).
    """
    return media_dir(media_key) / "index.json"


def frames_dir(media_key: str) -> Path:
    """Directory holding one extracted PNG per answered timestamp.

    USED BY: frame.py (writes), service.py (cache check), api.py (serves them).
    """
    return media_dir(media_key) / "frames"


def ensure_frames_dir(media_key: str) -> Path:
    """frames_dir(), creating it if absent.

    USED BY: frame.py, immediately before writing a PNG.
    """
    path = frames_dir(media_key)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Resolve cache -----------------------------------------------------------
# Keyed by URL rather than media_key, because the media_key is only known AFTER
# resolving. Lives outside the per-video dirs so it can never collide with one.

def resolve_cache_dir() -> Path:
    """Directory for cached yt-dlp resolve results.

    USED BY: resolve_cache_path, ensure_resolve_cache_dir.
    """
    return config.DATA_DIR / "_resolve"


def resolve_cache_path(url: str) -> Path:
    """Cache file for one URL's resolved metadata.

    WHY HASH THE URL: URLs contain characters that are illegal in filenames and
    can exceed path length limits. A truncated SHA-256 is short, safe, and
    stable for the same URL across runs.

    USED BY: resolve.py (load_cached_resolve, save_cached_resolve).
    """
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:24]
    return resolve_cache_dir() / f"{digest}.json"


def ensure_resolve_cache_dir() -> Path:
    """resolve_cache_dir(), creating it if absent.

    USED BY: resolve.py (save_cached_resolve).
    """
    path = resolve_cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
