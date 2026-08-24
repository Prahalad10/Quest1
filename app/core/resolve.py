"""Resolve a video URL to metadata and stream URLs -- metadata only, no download.

This is the only module that talks to yt-dlp. It answers five questions:

    1. What is this? (title, duration, extractor, id)
    2. Is it something we refuse? (playlist / live / DRM / too long)
    3. Where is the audio-only stream? (for ASR -- we never fetch the video body)
    4. Where is the video stream? (for a single ranged frame read, later)
    5. Does the host honour HTTP Range requests? (Step 5 depends on it)

The `media_key` it computes is derived from extractor + video id ONLY -- never
from the signed stream URL, which rotates on every resolve. That stability is
what makes the on-disk cache work across runs.

Run directly:
    python -m app.core.resolve "<video_url>"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from app import config
from app.errors import InvalidInputError, ResolveError, UnsupportedMediaError
from app.progress import ProgressCallback, report

STAGE = "resolve"

# Protocols we can do a ranged byte seek against. HLS/DASH manifests are not
# single seekable byte streams, so they are unusable for Step 5.
SEEKABLE_PROTOCOLS = {"http", "https"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class StreamChoice:
    """One selected format, flattened to the fields downstream steps need."""

    format_id: str
    url: str
    ext: Optional[str] = None
    protocol: Optional[str] = None
    vcodec: Optional[str] = None
    acodec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    abr: Optional[float] = None
    tbr: Optional[float] = None
    filesize: Optional[int] = None
    format_note: Optional[str] = None
    http_headers: dict[str, str] = field(default_factory=dict)
    expires_at: Optional[int] = None  # unix ts parsed from the signed URL, if present


@dataclass
class ResolvedMedia:
    """Everything Steps 2-6 need to know about a video, with no video bytes read."""

    media_key: str
    source_url: str
    extractor: str
    video_id: str
    title: str
    duration: float
    was_live: bool
    audio: StreamChoice
    video: StreamChoice
    has_separate_audio_stream: bool
    supports_http_ranges: bool
    range_check_detail: str
    resolved_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def compute_media_key(extractor: str, video_id: str) -> str:
    """Stable, filesystem-safe cache key.

    Human-readable prefix for debugging, plus a hash suffix so two ids that
    sanitize to the same slug can never collide.
    """
    raw = f"{extractor}:{video_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", f"{extractor}-{video_id}").strip("-").lower()
    return f"{slug[:48]}-{digest}"


def _parse_url_expiry(url: str) -> Optional[int]:
    """Signed CDN URLs carry an expiry; Step 5 uses it to decide when to re-resolve."""
    try:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    except ValueError:
        return None
    for key in ("expire", "expires", "Expires"):
        values = params.get(key)
        if values:
            try:
                return int(values[0])
            except (TypeError, ValueError):
                continue
    return None


def _validate_url(url: str) -> None:
    if not isinstance(url, str) or not url.strip():
        raise InvalidInputError("Video URL is required and must be a non-empty string.")
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise InvalidInputError(
            f"Video URL must start with http:// or https:// (got {url!r})."
        )
    if not parsed.netloc:
        raise InvalidInputError(f"Video URL has no host: {url!r}")


def _is_audio_only(fmt: dict[str, Any]) -> bool:
    return fmt.get("vcodec") in (None, "none") and fmt.get("acodec") not in (None, "none")


def _is_video_only(fmt: dict[str, Any]) -> bool:
    return fmt.get("vcodec") not in (None, "none") and fmt.get("acodec") in (None, "none")


def _has_video(fmt: dict[str, Any]) -> bool:
    return fmt.get("vcodec") not in (None, "none")


def _usable(fmt: dict[str, Any]) -> bool:
    return bool(fmt.get("url")) and not fmt.get("has_drm")


def _to_choice(fmt: dict[str, Any]) -> StreamChoice:
    url = fmt["url"]
    return StreamChoice(
        format_id=str(fmt.get("format_id", "?")),
        url=url,
        ext=fmt.get("ext"),
        protocol=fmt.get("protocol"),
        vcodec=fmt.get("vcodec"),
        acodec=fmt.get("acodec"),
        width=fmt.get("width"),
        height=fmt.get("height"),
        fps=fmt.get("fps"),
        abr=fmt.get("abr"),
        tbr=fmt.get("tbr"),
        filesize=fmt.get("filesize") or fmt.get("filesize_approx"),
        format_note=fmt.get("format_note"),
        http_headers=dict(fmt.get("http_headers") or {}),
        expires_at=_parse_url_expiry(url),
    )


# --------------------------------------------------------------------------- #
# Rejection rules
# --------------------------------------------------------------------------- #

def _reject_unsupported(info: dict[str, Any], url: str, max_duration: int) -> None:
    """Raise UnsupportedMediaError for anything this pipeline must not process."""
    if info.get("_type") == "playlist" or "entries" in info:
        count = len(info.get("entries") or [])
        raise UnsupportedMediaError(
            f"{url} is a playlist ({count} entries). Pass the URL of a single video."
        )

    live_status = info.get("live_status")
    if info.get("is_live") or live_status == "is_live":
        raise UnsupportedMediaError(
            f"{url} is a live stream. Live content has no fixed timeline to index."
        )
    if live_status == "is_upcoming":
        raise UnsupportedMediaError(f"{url} is an upcoming premiere and has no content yet.")
    if live_status == "post_live":
        raise UnsupportedMediaError(
            f"{url} just finished streaming and is still being processed by the host. "
            "Retry once the VOD is available."
        )

    formats = info.get("formats") or []
    if info.get("_has_drm") or (formats and all(f.get("has_drm") for f in formats)):
        raise UnsupportedMediaError(f"{url} is DRM-protected and cannot be processed.")

    duration = info.get("duration")
    if duration is None:
        raise UnsupportedMediaError(
            f"{url} reports no duration. Without a known duration the timeline cannot "
            "be validated, so this is refused rather than guessed."
        )
    try:
        duration = float(duration)
    except (TypeError, ValueError) as exc:
        raise UnsupportedMediaError(f"{url} reports a non-numeric duration: {duration!r}") from exc
    if duration <= 0:
        raise UnsupportedMediaError(f"{url} reports a non-positive duration: {duration}")
    if duration > max_duration:
        raise UnsupportedMediaError(
            f"{url} is {duration:.0f}s long, over the {max_duration}s cap. "
            "Raise QUEST1_MAX_DURATION to allow it."
        )


# --------------------------------------------------------------------------- #
# Format selection
# --------------------------------------------------------------------------- #

def select_audio_format(formats: list[dict[str, Any]]) -> dict[str, Any]:
    """Best audio-only stream: highest bitrate wins.

    Falls back to a progressive format only if the host offers no audio-only
    stream at all -- that costs video bytes, so it is a last resort.
    """
    candidates = [f for f in formats if _usable(f) and _is_audio_only(f)]
    if candidates:
        return max(candidates, key=lambda f: (f.get("abr") or f.get("tbr") or 0.0))

    progressive = [
        f for f in formats
        if _usable(f) and _has_video(f) and f.get("acodec") not in (None, "none")
    ]
    if progressive:
        return min(progressive, key=lambda f: (f.get("tbr") or float("inf")))

    raise UnsupportedMediaError(
        "No audio stream found in any available format. Cannot transcribe silent media."
    )


def select_video_format(formats: list[dict[str, Any]], max_height: int) -> dict[str, Any]:
    """Best seekable video stream at or below `max_height`.

    Prefers video-only (DASH) over progressive so a ranged read pulls no audio
    bytes, and requires a plain-HTTP protocol because Step 5 seeks by byte range.
    """
    seekable = [
        f for f in formats
        if _usable(f) and _has_video(f) and f.get("protocol") in SEEKABLE_PROTOCOLS
    ]
    if not seekable:
        raise UnsupportedMediaError(
            "No plain-HTTP video format available (only HLS/DASH manifests). "
            "Ranged frame extraction requires a directly seekable stream."
        )

    def rank(f: dict[str, Any]) -> tuple:
        height = f.get("height") or 0
        # Within the cap, taller is better; above it, prefer the least-oversized.
        within_cap = height <= max_height
        return (within_cap, height if within_cap else -height, f.get("tbr") or 0.0)

    video_only = [f for f in seekable if _is_video_only(f)]
    pool = video_only or seekable
    return max(pool, key=rank)


# --------------------------------------------------------------------------- #
# HTTP range probe
# --------------------------------------------------------------------------- #

def probe_http_ranges(
    url: str,
    headers: dict[str, str],
    timeout: int = config.NETWORK_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Ask the host for the first two bytes and see whether it honours Range.

    Returns (supported, human-readable detail). A failure here is reported, not
    raised: it is diagnostic information for Step 5, not a reason to abort.
    """
    request = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            accept_ranges = response.headers.get("Accept-Ranges", "")
            content_range = response.headers.get("Content-Range")
            if status == 206:
                return True, f"HTTP 206 Partial Content (Content-Range: {content_range})"
            if accept_ranges.lower() == "bytes":
                return True, f"HTTP {status} but Accept-Ranges: bytes"
            return False, (
                f"HTTP {status} with no partial-content support "
                f"(Accept-Ranges: {accept_ranges or 'absent'})"
            )
    except urllib.error.HTTPError as exc:
        return False, f"HTTP error {exc.code} {exc.reason}"
    except urllib.error.URLError as exc:
        return False, f"connection failed: {exc.reason}"
    except (TimeoutError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def resolve(
    url: str,
    *,
    max_duration: Optional[int] = None,
    max_height: Optional[int] = None,
    check_ranges: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> ResolvedMedia:
    """Extract metadata and stream URLs for `url` without downloading media.

    Raises InvalidInputError for a malformed URL, ResolveError if yt-dlp cannot
    extract the page, and UnsupportedMediaError for playlists, live streams,
    DRM, over-long videos, or an unseekable format set.
    """
    _validate_url(url)
    url = url.strip()
    max_duration = config.MAX_VIDEO_DURATION_SECONDS if max_duration is None else max_duration
    max_height = config.MAX_VIDEO_HEIGHT if max_height is None else max_height

    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError, ExtractorError
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ResolveError(
            "yt-dlp is not installed. Run: pip install -r requirements.txt"
        ) from exc

    report(progress_callback, STAGE, f"extracting metadata for {url}")

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,      # a watch?v=...&list=... URL resolves to the single video
        "extract_flat": False,
        "socket_timeout": config.NETWORK_TIMEOUT_SECONDS,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except (DownloadError, ExtractorError) as exc:
        raise ResolveError(f"yt-dlp could not resolve {url}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface the real cause, never guess past it
        raise ResolveError(f"Unexpected failure resolving {url}: {type(exc).__name__}: {exc}") from exc

    if not info:
        raise ResolveError(f"yt-dlp returned no metadata for {url}.")

    _reject_unsupported(info, url, max_duration)

    formats = info.get("formats") or []
    if not formats:
        raise ResolveError(f"yt-dlp returned no playable formats for {url}.")

    audio_fmt = select_audio_format(formats)
    video_fmt = select_video_format(formats, max_height)
    audio = _to_choice(audio_fmt)
    video = _to_choice(video_fmt)

    has_separate_audio = any(_usable(f) and _is_audio_only(f) for f in formats)
    report(
        progress_callback,
        STAGE,
        f"selected audio format {audio.format_id} ({audio.ext}, {audio.abr or '?'}kbps) "
        f"and video format {video.format_id} ({video.ext}, {video.height or '?'}p)",
    )

    if check_ranges:
        report(progress_callback, STAGE, "probing video host for HTTP Range support")
        supports_ranges, range_detail = probe_http_ranges(video.url, video.http_headers)
    else:
        supports_ranges, range_detail = False, "not checked (--skip-range-check)"

    extractor = str(info.get("extractor_key") or info.get("extractor") or "generic")
    video_id = str(info.get("id") or hashlib.sha256(url.encode()).hexdigest()[:16])

    resolved = ResolvedMedia(
        media_key=compute_media_key(extractor, video_id),
        source_url=url,
        extractor=extractor,
        video_id=video_id,
        title=str(info.get("title") or "(untitled)"),
        duration=float(info["duration"]),
        was_live=bool(info.get("was_live") or info.get("live_status") == "was_live"),
        audio=audio,
        video=video,
        has_separate_audio_stream=has_separate_audio,
        supports_http_ranges=supports_ranges,
        range_check_detail=range_detail,
        resolved_at=time.time(),
    )
    report(progress_callback, STAGE, f"resolved media_key={resolved.media_key}")
    return resolved


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.core.resolve",
        description="Resolve a video URL to metadata + stream URLs (no download).",
    )
    parser.add_argument("url", help="Video URL to resolve")
    parser.add_argument(
        "--max-duration", type=int, default=None,
        help=f"Duration cap in seconds (default {config.MAX_VIDEO_DURATION_SECONDS})",
    )
    parser.add_argument(
        "--max-height", type=int, default=None,
        help=f"Video height cap (default {config.MAX_VIDEO_HEIGHT})",
    )
    parser.add_argument(
        "--skip-range-check", action="store_true",
        help="Do not send a probe request to the video host",
    )
    parser.add_argument(
        "--no-urls", action="store_true",
        help="Redact the (very long, signed) stream URLs from the JSON output",
    )
    args = parser.parse_args(argv)

    try:
        resolved = resolve(
            args.url,
            max_duration=args.max_duration,
            max_height=args.max_height,
            check_ranges=not args.skip_range_check,
        )
    except (InvalidInputError, UnsupportedMediaError, ResolveError) as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    payload = resolved.to_dict()
    if args.no_urls:
        for key in ("audio", "video"):
            payload[key]["url"] = f"<redacted {len(payload[key]['url'])} chars>"
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
