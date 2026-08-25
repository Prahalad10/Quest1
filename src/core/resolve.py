"""yt-dlp metadata and stream selection. The ONLY module that imports yt-dlp.

Produces a ResolvedMedia: a media_key, plus one audio and one video stream to
work from. Nothing downstream ever sees a raw yt-dlp format dict.

Three things here are load-bearing and were each a real bug:

  * media_key derives from extractor + video id ONLY, never the stream URL,
    which is signed and rotates within hours.
  * Audio track selection ranks LANGUAGE above bitrate. Multi-language uploads
    carry dubs encoded higher than the original; bitrate alone transcribed an
    English video into Arabic.
  * Formats with unset codec fields still count. Some hosts (ok.ru) report
    neither vcodec nor acodec, and requiring known codecs discarded the only
    usable stream.

    python -m src.core.resolve <url>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from src import config, paths
from src.errors import InvalidInputError, DialogueFrameError, ResolveError, UnsupportedMediaError
from src.progress import ProgressCallback, report

STAGE = "resolve"

# HLS/DASH manifests are not single seekable byte streams, so a ranged frame
# seek against one is impossible.
SEEKABLE_PROTOCOLS = {"http", "https"}

# Used only when a host leaves the codec fields unset and the extension is the
# only evidence available.
VIDEO_CONTAINER_EXTS = frozenset({"mp4", "m4v", "webm", "mkv", "mov", "flv", "avi", "3gp", "ts"})


@dataclass
class StreamChoice:
    """One selected format, flattened to the fields downstream stages use.

    http_headers matters more than it looks: the CDN rejects requests whose
    User-Agent does not match the session the signed URL was issued to, so
    these must reach every ffmpeg call touching the URL.
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
    language: Optional[str] = None
    language_preference: Optional[int] = None
    # False means this format carries video we must fetch and discard.
    is_audio_only: bool = False
    http_headers: dict[str, str] = field(default_factory=dict)
    expires_at: Optional[int] = None


@dataclass
class ResolvedMedia:
    """What later stages need about a video, with no video bytes read.

    media_key and source_url are permanent; audio.url and video.url are signed
    and expire. That split is why the cache keys on the former and validates
    against the latter.
    """

    media_key: str
    source_url: str
    extractor: str
    video_id: str
    title: str
    duration: float
    audio: StreamChoice
    video: StreamChoice
    resolved_at: float
    # >1 means a multi-language upload, where the wrong track transcribes the
    # wrong language silently. audio.py uses it to judge pre-existing caches.
    audio_track_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedMedia":
        """asdict() flattens the nested StreamChoices, so rebuild them here."""
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        payload["audio"] = StreamChoice(**data["audio"])
        payload["video"] = StreamChoice(**data["video"])
        return cls(**payload)

    def stream_urls_expire_at(self) -> Optional[int]:
        """Earliest expiry of the two URLs -- the entry is only as good as that."""
        stamps = [s for s in (self.audio.expires_at, self.video.expires_at) if s]
        return min(stamps) if stamps else None


def slugify(text: str) -> str:
    """Fold arbitrary text into a filesystem-safe lowercase slug.

    Accents are stripped to ASCII rather than dropped, so "Pokémon" becomes
    "pokemon" and not "pokmon". Titles in a fully non-Latin script legitimately
    reduce to "" -- callers must have a fallback.
    """
    ascii_text = (unicodedata.normalize("NFKD", text or "")
                  .encode("ascii", "ignore").decode("ascii"))
    return re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower()


def media_key_digest(extractor: str, video_id: str) -> str:
    """The IDENTITY half of a media_key: 10 hex chars of extractor + video id.

    Derived from identity only, never the stream URL (which is signed and
    rotates) and never the title (which can be edited). This is what makes a
    key stable, so it must not be "tidied": the exact bytes hashed determine
    every path under outputs/.
    """
    return hashlib.sha256(f"{extractor}:{video_id}".encode("utf-8")).hexdigest()[:10]


def find_existing_media_key(digest: str) -> Optional[str]:
    """An existing outputs/ directory for this digest, whatever its title slug.

    Titles get edited. Without this, a renamed video would produce a new
    directory name, miss its own cache, and re-run a full ASR pass over audio
    already sitting on disk. The digest is the identity; the slug is only a
    label, so an existing directory carrying the right digest wins.
    """
    try:
        candidates = [d.name for d in config.OUTPUT_DIR.iterdir()
                      if d.is_dir() and d.name.endswith(f"-{digest}")]
    except OSError:
        return None
    return sorted(candidates)[0] if candidates else None


def compute_media_key(extractor: str, video_id: str, title: Optional[str] = None) -> str:
    """Cache key: a readable title slug plus a stable identity digest.

    e.g. "me-at-the-zoo-103eea2ce1"

    The slug exists so outputs/ is browsable -- it carries no meaning. The digest
    decides identity, so two videos sharing a title cannot collide and a
    retitled video keeps its cache (see find_existing_media_key).

    Falls back to the video id when a title is absent or slugifies to nothing,
    which is what happens for a title written entirely in a non-Latin script.
    """
    digest = media_key_digest(extractor, video_id)

    existing = find_existing_media_key(digest)
    if existing:
        return existing

    slug = slugify(title or "")
    if not slug:
        slug = slugify(f"{extractor}-{video_id}")
    return f"{slug[:60]}-{digest}"


def _parse_url_expiry(url: str) -> Optional[int]:
    """Signed URLs carry ?expire=<unix ts>; None means the host declares none."""
    try:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    except ValueError:
        return None
    for key in ("expire", "expires", "Expires"):
        if params.get(key):
            try:
                return int(params[key][0])
            except (TypeError, ValueError):
                continue
    return None


def _reject_playlist_url(url: str) -> None:
    """Refuse playlist URLs before yt-dlp silently reinterprets them.

    noplaylist=True makes yt-dlp resolve a playlist URL to some single video
    rather than erroring, which would answer a question the user did not ask.
    """
    parsed = urllib.parse.urlparse(url.strip())
    path = parsed.path.lower().rstrip("/")
    query = urllib.parse.parse_qs(parsed.query)
    addresses_playlist = path.endswith("/playlist") or path == "/playlist"
    list_without_video = "list" in query and "v" not in query and not path.endswith("/watch")
    if addresses_playlist or list_without_video:
        raise InvalidInputError(
            f"{url} addresses a playlist, not a single video. Pass one video URL."
        )


def _validate_url(url: str) -> None:
    if not isinstance(url, str) or not url.strip():
        raise InvalidInputError("A video URL is required.")
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise InvalidInputError(f"Video URL must start with http:// or https:// (got {url!r}).")
    if not parsed.netloc:
        raise InvalidInputError(f"Video URL has no host: {url!r}")
    _reject_playlist_url(url)


# --- Format classification ---------------------------------------------------
# Hosts that report no codec information at all broke all three of these, each
# in its own way. A container with BOTH codec fields unset is assumed to carry
# what its extension implies; a stream that reports a real acodec can never be
# caught by that guess.

def _is_audio_only(fmt: dict[str, Any]) -> bool:
    return fmt.get("vcodec") in (None, "none") and fmt.get("acodec") not in (None, "none")


def _is_video_only(fmt: dict[str, Any]) -> bool:
    return fmt.get("vcodec") not in (None, "none") and fmt.get("acodec") in (None, "none")


def _has_video(fmt: dict[str, Any]) -> bool:
    if fmt.get("vcodec") not in (None, "none"):
        return True
    if fmt.get("vcodec") is None and fmt.get("acodec") is None:
        return (fmt.get("ext") or "").lower() in VIDEO_CONTAINER_EXTS
    return False


def _may_have_audio(fmt: dict[str, Any]) -> bool:
    if fmt.get("acodec") not in (None, "none"):
        return True
    return (fmt.get("acodec") is None and fmt.get("vcodec") is None
            and (fmt.get("ext") or "").lower() in VIDEO_CONTAINER_EXTS)


def _usable(fmt: dict[str, Any]) -> bool:
    return bool(fmt.get("url")) and not fmt.get("has_drm")


def _to_choice(fmt: dict[str, Any]) -> StreamChoice:
    """The boundary where yt-dlp's schema stops and ours begins."""
    return StreamChoice(
        format_id=str(fmt.get("format_id", "?")),
        url=fmt["url"],
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
        is_audio_only=_is_audio_only(fmt),
        http_headers=dict(fmt.get("http_headers") or {}),
        expires_at=_parse_url_expiry(fmt["url"]),
    )


def _reject_unsupported(info: dict[str, Any], url: str, max_duration: int) -> None:
    """Every refusal lives here so the rules are auditable in one place.

    A missing duration is refused rather than tolerated: the audio truncation
    check and the frame clamp both become meaningless without it.
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
            f"{url} reports no duration. Without one the timeline cannot be validated, "
            "so this is refused rather than guessed."
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
            "Raise DIALOGUEFRAME_MAX_DURATION to allow it."
        )


def count_audio_languages(formats: list[dict[str, Any]]) -> int:
    """Distinct audio LANGUAGES, not formats.

    YouTube publishes the same audio in several codecs, so counting formats
    returns 5+ for an ordinary single-language video and would mark every
    cached wav stale.
    """
    languages = {(f.get("language") or "").lower()
                 for f in formats if _usable(f) and _is_audio_only(f)}
    languages.discard("")
    return max(1, len(languages))


def select_audio_format(
    formats: list[dict[str, Any]], prefer_language: Optional[str] = None
) -> dict[str, Any]:
    """Original language first, then bitrate.

    Ranking on bitrate alone picked an Arabic dub of an English video, because
    dubs are often encoded higher than the original. yt-dlp marks the original
    language_preference=10 and every dub -1.

    Falls back to the SMALLEST progressive format when a host offers no
    audio-only track at all -- the one path that fetches video bytes.
    """
    def rank(fmt: dict[str, Any]) -> tuple:
        language = (fmt.get("language") or "").lower()
        requested = 1 if (prefer_language and language == prefer_language.lower()) else 0
        preference = fmt.get("language_preference")
        return (requested, -1 if preference is None else int(preference),
                fmt.get("abr") or fmt.get("tbr") or 0.0)

    candidates = [f for f in formats if _usable(f) and _is_audio_only(f)]
    if candidates:
        return max(candidates, key=rank)

    progressive = [(i, f) for i, f in enumerate(formats)
                   if _usable(f) and _has_video(f) and _may_have_audio(f)
                   and f.get("protocol") in SEEKABLE_PROTOCOLS]
    if progressive:
        # Smallest first: every byte here is video being thrown away. When a
        # host reports no bitrate or height, yt-dlp's own worst-first ordering
        # is the only signal left.
        return min(progressive,
                   key=lambda p: (p[1].get("tbr") or float("inf"),
                                  p[1].get("height") or float("inf"), p[0]))[1]

    raise UnsupportedMediaError(
        "No audio stream found in any available format. Cannot transcribe silent media."
    )


def select_video_format(formats: list[dict[str, Any]], max_height: int) -> dict[str, Any]:
    """Best seekable video at or below the height cap.

    Prefers video-only (DASH) so a ranged read pulls no audio bytes. The index
    tiebreak is not cosmetic: hosts that report neither height nor bitrate made
    every candidate tie, and max() then returned yt-dlp's worst-quality-first
    entry, extracting frames from `mobile` while `hd` sat unused.
    """
    seekable = [(i, f) for i, f in enumerate(formats)
                if _usable(f) and _has_video(f) and f.get("protocol") in SEEKABLE_PROTOCOLS]
    if not seekable:
        raise UnsupportedMediaError(
            "No plain-HTTP video format available (only HLS/DASH manifests). "
            "Ranged frame extraction requires a directly seekable stream."
        )

    def rank(pair: tuple[int, dict[str, Any]]) -> tuple:
        index, fmt = pair
        height = fmt.get("height") or 0
        within_cap = height <= max_height
        # Above the cap, negative height picks the least-oversized option.
        return (within_cap, height if within_cap else -height, fmt.get("tbr") or 0.0, index)

    video_only = [p for p in seekable if _is_video_only(p[1])]
    return max(video_only or seekable, key=rank)[1]


# --- Transient failure handling ----------------------------------------------
# ok.ru resets a share of connections: the same URL fails, succeeds, then fails
# again with nothing changed. yt-dlp flattens socket errors into DownloadError,
# so the exception type carries no useful distinction and the message is all
# there is to match on.

TRANSIENT_ERROR_MARKERS = (
    "forcibly closed", "connection reset", "connection aborted", "timed out",
    "temporary failure", "eof occurred", "remote end closed",
    "unable to download webpage", "read operation timed out",
    "connection refused", "bad gateway", "service unavailable",
)

# Stable answers, never retried. Keep each specific enough that it cannot match
# inside another word: a bare "age" matched "unable to download webPAGE".
PERMANENT_ERROR_MARKERS = (
    "is private", "video unavailable", "has been removed", "not available",
    "unsupported url", "drm", "members-only", "sign in to confirm",
    "age-restricted", "age restricted", "requested format is not available",
    "http error 404", "http error 403", "http error 410",
)


def is_transient_error(exc: Exception) -> bool:
    """True when a failure looks like a flaky transport rather than a refusal."""
    message = str(exc).lower()
    if any(m in message for m in PERMANENT_ERROR_MARKERS):
        return False
    return any(m in message for m in TRANSIENT_ERROR_MARKERS)


def retry_transient(
    operation: Callable[[], Any],
    *,
    description: str,
    stage: str,
    progress_callback: Optional[ProgressCallback] = None,
    attempts: Optional[int] = None,
    backoff: Optional[float] = None,
) -> Any:
    """Call operation(), retrying only transport-shaped failures.

    Shared by resolve() and the audio download because BOTH run the yt-dlp
    extractor; retrying in one place alone just moved the failure one stage
    later. Re-raises the last exception unchanged so callers keep their wording.
    """
    attempts = max(1, attempts if attempts is not None else config.RESOLVE_MAX_ATTEMPTS)
    backoff = backoff if backoff is not None else config.RESOLVE_RETRY_BACKOFF_SECONDS

    for attempt in range(1, attempts + 1):
        try:
            result = operation()
            if attempt > 1:
                report(progress_callback, stage,
                       f"{description} succeeded on attempt {attempt}/{attempts}")
            return result
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            if attempt >= attempts or not is_transient_error(exc):
                raise
            delay = backoff * attempt
            report(progress_callback, stage,
                   f"transient network failure during {description} "
                   f"(attempt {attempt}/{attempts}); retrying in {delay:.1f}s")
            time.sleep(delay)


# --- Resolve cache -----------------------------------------------------------

def _cached_resolve_is_usable(payload: dict[str, Any]) -> bool:
    """Valid only while the signed URLs are, AND only if its key still derives.

    Prefers the URL's declared expiry, falling back to a conservative age limit.
    Erring permissive means a 403 mid-pipeline; erring strict costs one
    re-resolve, so the fallback errs strict.

    THE DIGEST CHECK IS NOT REDUNDANT. media_key is stored in the entry, and
    everything under outputs/ is addressed by it. When the key scheme changed,
    stale entries kept returning the OLD key, so the pipeline looked in an
    empty directory and silently re-ran ASR against a cache sitting right
    there -- while reporting a cache HIT. Only the digest is compared: the
    title slug is a label and is allowed to drift.
    """
    try:
        media = ResolvedMedia.from_dict(payload)
    except (KeyError, TypeError):
        return False

    if not media.media_key.endswith("-" + media_key_digest(media.extractor, media.video_id)):
        return False

    expires_at = media.stream_urls_expire_at()
    if expires_at:
        return time.time() < (expires_at - config.FRAME_URL_EXPIRY_MARGIN_SECONDS)
    return (time.time() - media.resolved_at) < config.RESOLVE_CACHE_TTL_SECONDS


def load_cached_resolve(url: str) -> Optional[ResolvedMedia]:
    """A corrupt entry returns None rather than raising: re-resolving costs
    seconds and can never produce a wrong answer."""
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
    """Atomic, so a concurrent reader never sees a half-written document."""
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
    """resolve(), reusing a cached result while its stream URLs are valid.

    Returns (media, was_cached). This is what makes a repeat query fast once
    ASR is cached -- a fresh extraction would otherwise dominate the response.
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
        report(progress_callback, STAGE, f"WARNING: could not write resolve cache: {exc}")
    return media, False


def resolve(
    url: str,
    *,
    max_duration: Optional[int] = None,
    max_height: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> ResolvedMedia:
    """Extract metadata and pick the audio and video streams to work from."""
    _validate_url(url)
    url = url.strip()
    max_duration = config.MAX_VIDEO_DURATION_SECONDS if max_duration is None else max_duration
    max_height = config.MAX_VIDEO_HEIGHT if max_height is None else max_height

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ResolveError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

    report(progress_callback, STAGE, f"extracting metadata for {url}")
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
        "socket_timeout": config.NETWORK_TIMEOUT_SECONDS,
        "retries": config.RESOLVE_HTTP_RETRIES,
        "extractor_retries": config.RESOLVE_HTTP_RETRIES,
    }

    def extract() -> dict[str, Any]:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            raise ResolveError(f"yt-dlp returned no metadata for {url}.")
        return info

    try:
        info = retry_transient(extract, description="metadata extraction", stage=STAGE,
                               progress_callback=progress_callback)
    except DialogueFrameError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface the real cause
        raise ResolveError(f"yt-dlp could not resolve {url}: {exc}") from exc

    _reject_unsupported(info, url, max_duration)
    formats = info.get("formats") or []
    if not formats:
        raise ResolveError(f"yt-dlp returned no playable formats for {url}.")

    audio = _to_choice(select_audio_format(formats, config.AUDIO_TRACK_LANGUAGE))
    video = _to_choice(select_video_format(formats, max_height))

    report(progress_callback, STAGE,
           f"selected audio {audio.format_id} ({audio.ext}, {audio.abr or '?'}kbps, "
           f"lang={audio.language or 'n/a'}) and video {video.format_id} "
           f"({video.ext}, {video.height or '?'}p)")

    track_count = count_audio_languages(formats)
    if track_count > 1:
        report(progress_callback, STAGE,
               f"{track_count} audio languages available; using {audio.language or 'original'} "
               f"(set DIALOGUEFRAME_AUDIO_LANGUAGE to choose another)")

    extractor = str(info.get("extractor_key") or info.get("extractor") or "unknown")
    video_id = str(info.get("id") or "")
    title = str(info.get("title") or "(untitled)")
    media = ResolvedMedia(
        media_key=compute_media_key(extractor, video_id, title),
        source_url=url,
        extractor=extractor,
        video_id=video_id,
        title=title,
        duration=float(info["duration"]),
        audio=audio,
        video=video,
        resolved_at=time.time(),
        audio_track_count=track_count,
    )
    report(progress_callback, STAGE, f"resolved media_key={media.media_key}")
    return media


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.core.resolve")
    parser.add_argument("url")
    parser.add_argument("--force", action="store_true", help="Ignore the resolve cache")
    args = parser.parse_args(argv)
    try:
        media, cached = resolve_cached(args.url, force=args.force)
    except DialogueFrameError as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2
    print(media.to_json())
    print(f"(cached: {cached})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
