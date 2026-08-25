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
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from app import config, paths
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
    """One selected format, flattened to the fields downstream steps need.

    yt-dlp format dicts carry dozens of keys; this keeps only what audio.py and
    frame.py actually use, so those modules never have to know yt-dlp schema.

    `http_headers` matters more than it looks: the CDN rejects requests whose
    User-Agent does not match the session the signed URL was issued for, so
    these must be passed to every ffmpeg invocation touching the URL.

    USED BY: ResolvedMedia (audio and video slots).
    """

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
    # Which audio track this is. On a multi-language upload these decide whether
    # we transcribe the original or a dub -- see select_audio_format.
    language: Optional[str] = None
    language_preference: Optional[int] = None
    http_headers: dict[str, str] = field(default_factory=dict)
    expires_at: Optional[int] = None  # unix ts parsed from the signed URL, if present


@dataclass
class ResolvedMedia:
    """Everything later stages need about a video, with no video bytes read.

    Produced by resolve(), cached to disk by save_cached_resolve(), and passed
    down through audio.py, index.py and frame.py.

    Note what is and is not stable: `media_key` and `source_url` are permanent,
    while `audio.url` and `video.url` are signed and expire within hours. That
    split is why the cache is keyed on the former and validated against the
    latter.

    USED BY: app/service.py and every core module below it.
    """

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
    # How many distinct audio-only tracks the source offered. >1 means this is a
    # multi-language upload, where picking the wrong track transcribes the wrong
    # language perfectly and silently. Used by audio.py to decide whether a wav
    # cached before track selection existed can still be trusted.
    audio_track_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form. USED BY: to_json and the __main__ JSON output."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """JSON text. USED BY: save_cached_resolve and the __main__ block."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedMedia":
        """Rebuild from the dict form, restoring the nested StreamChoice objects.

        asdict() flattens the dataclasses to plain dicts, so a naive cls(**data)
        would leave `audio` and `video` as dicts and every attribute access on
        them would fail. This is the inverse that keeps the round-trip honest.

        USED BY: load_cached_resolve and _cached_resolve_is_usable.
        """
        payload = dict(data)
        payload["audio"] = StreamChoice(**data["audio"])
        payload["video"] = StreamChoice(**data["video"])
        return cls(**payload)

    def stream_urls_expire_at(self) -> Optional[int]:
        """Earliest expiry across the two signed URLs, if either declares one.

        The EARLIEST is used because the cached entry is only as good as its
        soonest-expiring member; reusing it past that point would produce a 403
        on whichever stream expired first.

        USED BY: _cached_resolve_is_usable.
        """
        stamps = [s for s in (self.audio.expires_at, self.video.expires_at) if s]
        return min(stamps) if stamps else None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def compute_media_key(extractor: str, video_id: str) -> str:
    """Stable, filesystem-safe cache key for one video.

    THE KEY DECISION IN THE CACHING DESIGN: derived from extractor + video id
    ONLY -- never from the stream URL. Signed URLs rotate on every resolve, so a
    URL-derived key would miss the cache every single time and re-run ASR on
    every query.

    Format: readable slug + hash suffix, e.g. "youtube-jnqxac9ivrw-103eea2ce1".
    The slug aids debugging; the hash guarantees two ids that sanitize to the
    same slug cannot collide.

    USED BY: resolve(), and therefore every path under data/{media_key}/.
    """
    raw = f"{extractor}:{video_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", f"{extractor}-{video_id}").strip("-").lower()
    return f"{slug[:48]}-{digest}"


def _parse_url_expiry(url: str) -> Optional[int]:
    """Extract the expiry timestamp from a signed CDN URL, if it has one.

    YouTube URLs carry ?expire=<unix ts>. Knowing it lets the resolve cache and
    frame.py refresh proactively instead of discovering the expiry through a
    failed request.

    Returns None for hosts that do not advertise expiry, in which case callers
    fall back to a conservative TTL.

    USED BY: _to_choice, for both the audio and video streams.
    """
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


def _reject_playlist_url(url: str) -> None:
    """Refuse a URL that addresses a PLAYLIST rather than a single video.

    WHY THIS IS NEEDED SEPARATELY FROM THE PLAYLIST CHECK IN _reject_unsupported:
    we pass noplaylist=True to yt-dlp, which makes it silently REINTERPRET a
    playlist URL as that playlist's first video. So the _type == "playlist"
    branch downstream almost never fires -- yt-dlp has already replaced the
    request with a different one. The user asked about a playlist and would get
    an answer about some video they never named, which is exactly the kind of
    silent fallback this project refuses to make.

    Deliberately narrow, so the common case keeps working:
        REJECTED  youtube.com/playlist?list=PL...     addresses a playlist
        ALLOWED   youtube.com/watch?v=X&list=PL...    addresses video X, which
                                                      happens to sit in a list

    USED BY: _validate_url, so both the CLI and the API refuse it identically.
    """
    parsed = urllib.parse.urlparse(url.strip())
    path = parsed.path.lower().rstrip("/")
    query = urllib.parse.parse_qs(parsed.query)

    addresses_playlist = path.endswith("/playlist") or path == "/playlist"
    # A bare list= with no video id and no /watch path is also a playlist
    # reference on every host that uses this convention.
    list_without_video = (
        "list" in query and "v" not in query and not path.endswith("/watch")
    )

    if addresses_playlist or list_without_video:
        raise InvalidInputError(
            f"{url} addresses a playlist, not a single video. Pass the URL of one "
            f"video instead -- e.g. the watch?v=... link for the specific video you "
            f"mean. (A watch?v=...&list=... URL is fine: it names a single video.)"
        )


def _validate_url(url: str) -> None:
    """Reject anything that is not a plausible http(s) URL, before any network use.

    WHY UP FRONT: yt-dlp given a bare string will try a series of extractors and
    eventually fail with a message about unsupported URLs, which tells the user
    nothing useful. Catching it here produces one clear sentence instead.

    Also rejects playlist URLs -- see _reject_playlist_url for why that cannot be
    left to the downstream check.

    USED BY: resolve() and resolve_cached().
    """
    if not isinstance(url, str) or not url.strip():
        raise InvalidInputError("Video URL is required and must be a non-empty string.")
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise InvalidInputError(
            f"Video URL must start with http:// or https:// (got {url!r})."
        )
    if not parsed.netloc:
        raise InvalidInputError(f"Video URL has no host: {url!r}")
    _reject_playlist_url(url)


def _is_audio_only(fmt: dict[str, Any]) -> bool:
    """True for a format with audio and no video -- what ASR wants.

    USED BY: select_audio_format and the has_separate_audio_stream flag.
    """
    return fmt.get("vcodec") in (None, "none") and fmt.get("acodec") not in (None, "none")


def _is_video_only(fmt: dict[str, Any]) -> bool:
    """True for a DASH video-only format -- preferred for frame extraction.

    Preferred because a ranged read against it pulls no audio bytes at all.

    USED BY: select_video_format.
    """
    return fmt.get("vcodec") not in (None, "none") and fmt.get("acodec") in (None, "none")


# Containers that carry video. Used only when a host leaves the codec fields
# unset, so the extension is the only evidence available.
VIDEO_CONTAINER_EXTS = frozenset({
    "mp4", "m4v", "webm", "mkv", "mov", "flv", "avi", "3gp", "ts",
})


def _has_video(fmt: dict[str, Any]) -> bool:
    """True for any format carrying a video stream, progressive or video-only.

    THE UNKNOWN-CODEC CASE IS NOT A CORNER CASE. Some extractors report neither
    vcodec nor acodec for their progressive stream -- ok.ru is one. Requiring a
    known vcodec discarded that stream, and since every other format the host
    offered was HLS or DASH, resolve then reported "no plain-HTTP video format
    available" for a video whose progressive MP4 was sitting right there and was
    perfectly seekable.

    A format with BOTH codec fields unset and a video container extension is
    therefore treated as carrying video. This cannot misclassify an audio-only
    stream: those report a real acodec, which excludes them from this branch.

    USED BY: select_video_format and select_audio_format.
    """
    if fmt.get("vcodec") not in (None, "none"):
        return True
    if fmt.get("vcodec") is None and fmt.get("acodec") is None:
        return (fmt.get("ext") or "").lower() in VIDEO_CONTAINER_EXTS
    return False


def _usable(fmt: dict[str, Any]) -> bool:
    """True when a format has a URL and is not individually DRM-protected.

    USED BY: both format selectors, as a common pre-filter.
    """
    return bool(fmt.get("url")) and not fmt.get("has_drm")


def _to_choice(fmt: dict[str, Any]) -> StreamChoice:
    """Convert a yt-dlp format dict into our own StreamChoice.

    The boundary where yt-dlp's schema stops and ours begins: nothing downstream
    of resolve() ever sees a raw yt-dlp dict.

    USED BY: resolve(), for the two selected formats.
    """
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
        language=fmt.get("language"),
        language_preference=fmt.get("language_preference"),
        http_headers=dict(fmt.get("http_headers") or {}),
        expires_at=_parse_url_expiry(url),
    )


# --------------------------------------------------------------------------- #
# Rejection rules
# --------------------------------------------------------------------------- #

def _reject_unsupported(info: dict[str, Any], url: str, max_duration: int) -> None:
    """Raise UnsupportedMediaError for anything this pipeline must not process.

    ALL REFUSALS LIVE HERE so the rules are auditable in one place: playlists,
    live streams, upcoming premieres, still-processing VODs, DRM, and videos
    with a missing, non-numeric, non-positive or over-cap duration.

    A missing duration is refused rather than tolerated because every later stage
    validates against it -- the audio truncation check and the frame clamp both
    become meaningless without it.

    USED BY: resolve(), immediately after extraction and before any format work.
    """
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

def _track_rank(fmt: dict[str, Any], prefer_language: Optional[str]) -> tuple:
    """Sort key for choosing between audio tracks. Higher is better.

    ORDER OF PRECEDENCE
      1. An explicitly requested language (QUEST1_AUDIO_LANGUAGE), if it exists.
      2. yt-dlp's language_preference, which marks the ORIGINAL/default track.
      3. Bitrate.

    WHY LANGUAGE OUTRANKS BITRATE: a multi-language upload carries the original
    plus a set of dubs, and the dubs are frequently encoded at a HIGHER bitrate
    than the original. Sorting on bitrate alone therefore picks a dub more or
    less at random. See select_audio_format for what that cost us.

    USED BY: select_audio_format.
    """
    language = (fmt.get("language") or "").lower()
    requested = 1 if (prefer_language and language == prefer_language.lower()) else 0
    # yt-dlp marks the original/default track 10 and every dub -1. Absent on
    # single-track uploads, where the value is the same for all candidates and
    # so cannot change the outcome.
    preference = fmt.get("language_preference")
    preference = -1 if preference is None else int(preference)
    bitrate = fmt.get("abr") or fmt.get("tbr") or 0.0
    return (requested, preference, bitrate)



def count_audio_languages(formats: list[dict[str, Any]]) -> int:
    """How many distinct audio LANGUAGES the source offers.

    *** COUNT LANGUAGES, NOT FORMATS. *** YouTube publishes the same audio in
    several codecs and bitrates (139, 140, 249, 250, 251...), so counting
    audio-only FORMATS returns five-plus for an ordinary single-language video.
    An earlier version did exactly that, decided every cached wav was from a
    multi-track upload, and re-fetched and re-transcribed every video that was
    already correctly cached.

    Returns 1 when the source labels no languages at all, which is the common
    case outside YouTube and means the question does not arise.

    USED BY: resolve(), to set ResolvedMedia.audio_track_count, which audio.py
    uses to decide whether a wav cached before track selection can be trusted.
    """
    languages = {
        (f.get("language") or "").lower()
        for f in formats if _usable(f) and _is_audio_only(f)
    }
    languages.discard("")
    return max(1, len(languages))


def select_audio_format(
    formats: list[dict[str, Any]], prefer_language: Optional[str] = None
) -> dict[str, Any]:
    """Best audio-only stream: ORIGINAL LANGUAGE first, then highest bitrate.

    *** WHY LANGUAGE COMES FIRST -- THIS WAS A REAL BUG ***
    A MrBeast upload carried fourteen audio tracks: one English original
    (language_preference=10) and thirteen dubs (-1), several of the dubs at a
    higher bitrate. Selecting on bitrate alone chose the ARABIC dub, so the
    whole video was transcribed into Arabic and every English query returned
    "not found" -- with no error anywhere, because nothing had actually failed.

    Set QUEST1_AUDIO_LANGUAGE to deliberately transcribe a specific dub; it is
    honoured only if a track in that language exists, otherwise the original is
    used rather than silently substituting something else.

    Falls back to a progressive format only if the host offers NO audio-only
    stream -- and then picks the LOWEST bitrate one, since every byte of that
    format is video we do not want. This is the one path that violates the
    "never fetch video" goal, so it is a genuine last resort and ensure_audio
    warns when it is taken.

    USED BY: resolve().
    """
    candidates = [f for f in formats if _usable(f) and _is_audio_only(f)]
    if candidates:
        return max(candidates, key=lambda f: _track_rank(f, prefer_language))

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

    THE PROTOCOL FILTER IS LOAD-BEARING: HLS and DASH manifest formats are not
    single seekable byte streams, so a ranged seek against one is impossible.
    Rejecting them here means frame extraction fails at RESOLVE time with a clear
    explanation, rather than deep inside ffmpeg after ASR has already run.

    Prefers video-only (DASH) over progressive so a ranged read pulls no audio
    bytes. The height cap keeps the read small: 4K buys nothing for a single PNG.

    USED BY: resolve().
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
        """Sort key: prefer within the cap, then taller, then higher bitrate.

        Formats above the cap sort by NEGATIVE height so that, if every option
        exceeds it, the least oversized one wins rather than the largest.
        """
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

    Verifies the assumption the whole project rests on. A 206 Partial Content
    response proves the host will serve byte ranges, which is what makes single-
    frame extraction possible without downloading the video.

    Returns (supported, human-readable detail) and NEVER raises: this is
    diagnostic information, and a probe that fails for an unrelated reason should
    not block a resolve whose other output is perfectly good.

    USED BY: resolve() when check_ranges is True. app/service.py passes False,
    since by then the answer would not change what it does.
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
# Resolve cache
# --------------------------------------------------------------------------- #

def _cached_resolve_is_usable(payload: dict[str, Any]) -> bool:
    """A cached resolve is only usable while its signed stream URLs still are.

    Two-tier check: prefer the URL's own declared expiry, and fall back to a
    conservative age limit for hosts that declare none. Getting this wrong in the
    permissive direction means a 403 mid-pipeline; in the strict direction it
    just costs an unnecessary re-resolve, so the fallback errs strict.

    USED BY: load_cached_resolve.
    """
    try:
        media = ResolvedMedia.from_dict(payload)
    except (KeyError, TypeError):
        return False

    expires_at = media.stream_urls_expire_at()
    now = time.time()
    if expires_at:
        return now < (expires_at - config.FRAME_URL_EXPIRY_MARGIN_SECONDS)
    # No declared expiry: fall back to a conservative age limit.
    return (now - media.resolved_at) < config.RESOLVE_CACHE_TTL_SECONDS


def load_cached_resolve(url: str) -> Optional[ResolvedMedia]:
    """Return a still-valid cached resolve for `url`, or None.

    A corrupt or expired entry is simply IGNORED here, unlike the transcript
    index which raises on corruption. The asymmetry is deliberate: re-resolving
    costs a couple of seconds and can never produce a wrong answer, whereas
    silently rebuilding a corrupt index would hide a real problem and cost
    minutes of ASR.

    USED BY: resolve_cached.
    """
    path = paths.resolve_cache_path(url)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not _cached_resolve_is_usable(payload):
        return None
    try:
        return ResolvedMedia.from_dict(payload)
    except (KeyError, TypeError):
        return None


def save_cached_resolve(media: ResolvedMedia) -> None:
    """Persist a resolve result so the next query on this URL can skip yt-dlp.

    Written atomically via a .part file, so a concurrent reader never sees a
    half-written document.

    USED BY: resolve_cached, after a successful fresh resolve.
    """
    paths.ensure_resolve_cache_dir()
    path = paths.resolve_cache_path(media.source_url)
    temp = path.with_suffix(".part")
    temp.write_text(media.to_json(), encoding="utf-8")
    os.replace(temp, path)


def resolve_cached(
    url: str,
    *,
    force: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    **kwargs: Any,
) -> tuple[ResolvedMedia, bool]:
    """resolve(), but reusing a cached result while its stream URLs are valid.

    Returns (media, from_cache). This is what makes a repeat query on the same
    video fast: a fresh yt-dlp extraction costs seconds and would otherwise
    dominate the entire response time once ASR is cached.

    A cache WRITE failure is reported and swallowed -- failing the whole request
    because a cache could not be saved would be the wrong trade.

    USED BY: app/service.py (stage 1). Prefer this over resolve() anywhere a
    repeat call is plausible.
    """
    _validate_url(url)
    url = url.strip()
    if not force:
        cached = load_cached_resolve(url)
        if cached is not None:
            report(progress_callback, STAGE, f"cache hit: media_key={cached.media_key}")
            return cached, True

    media = resolve(url, progress_callback=progress_callback, **kwargs)
    try:
        save_cached_resolve(media)
    except OSError as exc:
        # A cache write failure must not fail the request.
        report(progress_callback, STAGE, f"WARNING: could not write resolve cache: {exc}")
    return media, False


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

    The ONLY function in the project that talks to yt-dlp. Everything downstream
    consumes ResolvedMedia, so a yt-dlp API change touches this file alone.

    Always performs a fresh network extraction -- call resolve_cached() unless
    you specifically need current URLs.

    Raises InvalidInputError for a malformed URL, ResolveError if yt-dlp cannot
    extract the page, and UnsupportedMediaError for playlists, live streams,
    DRM, over-long videos, or an unseekable format set.

    USED BY: resolve_cached, and frame.py when retrying after an expired URL.
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

    audio_fmt = select_audio_format(formats, config.AUDIO_TRACK_LANGUAGE)
    video_fmt = select_video_format(formats, max_height)
    audio = _to_choice(audio_fmt)
    video = _to_choice(video_fmt)

    has_separate_audio = any(_usable(f) and _is_audio_only(f) for f in formats)
    report(
        progress_callback,
        STAGE,
        f"selected audio format {audio.format_id} "
        f"({audio.ext}, {audio.abr or '?'}kbps, lang={audio.language or 'n/a'}) "
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
        audio_track_count=count_audio_languages(formats),
    )
    if resolved.audio_track_count > 1:
        # Worth saying out loud: on these uploads the choice of track decides
        # what language the transcript comes out in.
        report(
            progress_callback, STAGE,
            f"{resolved.audio_track_count} audio tracks available; "
            f"using {audio.language or 'original'} "
            f"(set QUEST1_AUDIO_LANGUAGE to choose another)",
        )
    report(progress_callback, STAGE, f"resolved media_key={resolved.media_key}")
    return resolved


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry point: resolve a URL and print its metadata as JSON.

    WHY: the fastest way to check whether a video is usable before spending time
    on it. The three fields that matter are supports_http_ranges (frame
    extraction viability), audio.vcodec == "none" (a true audio-only stream
    exists), and video.protocol (must be https, not a manifest).

    Progress goes to stderr and JSON to stdout, so the output pipes cleanly.

    USED BY: `python -m app.core.resolve "<url>" [--no-urls]`.
    """
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
