"""Turn a timestamp into a frame number and a PNG, via a ranged HTTP seek.

The video is never downloaded. ffmpeg seeks twice: a coarse -ss BEFORE -i jumps
to the nearest keyframe using byte ranges, reading almost nothing, then a fine
-ss AFTER -i decodes forward to land on the exact frame. That is the difference
between a few hundred KB and a full download.

    python -m app.core.frame <media_key> <seconds>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from app import config, paths
from app.core import ffmpeg
from app.core.resolve import ResolvedMedia, resolve
from app.errors import FFmpegError, InvalidInputError, Quest1Error
from app.progress import ProgressCallback, report

STAGE = "frame"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class FrameError(Quest1Error):
    """Raised only after a re-resolve and retry both failed, or ffmpeg reported
    success but produced something that is not a valid PNG."""


@dataclass
class FrameResult:
    """`note` and `frame_number` work together: whenever frame_number is None,
    note explains why in a sentence meant for the user. Never show one without
    the other."""

    path: str
    timestamp_seconds: float
    frame_number: Optional[int]   # None for VFR -- see note
    frame_pts: Optional[float]
    fps: Optional[float]
    is_vfr: bool
    width: Optional[int]
    height: Optional[int]
    size_bytes: int
    note: Optional[str]
    re_resolved: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frame_number_for_timestamp(
    probe: dict[str, Any], seconds: float
) -> tuple[Optional[int], Optional[float], Optional[str]]:
    """(frame_number, frame_pts, note) for `seconds`. Pure arithmetic.

    For constant frame rate, frame N covers [N/fps, (N+1)/fps), so the frame on
    screen at t is floor(t * fps) -- floor, not round.

    For VFR the mapping is not a multiplication at all, since frame durations
    vary across the file. Returns None plus an explanation rather than guessing:
    a confidently wrong frame number is worse than an honest absence. The
    extracted IMAGE is correct either way; only the integer is unavailable.
    """
    if seconds < 0:
        raise InvalidInputError(f"Timestamp must be non-negative, got {seconds}.")

    if probe.get("is_vfr"):
        return None, None, (
            f"Source is variable-frame-rate (r_frame_rate {probe.get('r_frame_rate')} != "
            f"avg_frame_rate {probe.get('avg_frame_rate')}), so a frame number cannot be "
            "derived from the timestamp by arithmetic. The extracted image is still the "
            "correct frame for this timestamp."
        )

    fps = probe.get("fps")
    if not fps or fps <= 0:
        return None, None, (
            "Frame rate is unknown for this stream, so no frame number can be computed. "
            "The extracted image is still correct for this timestamp."
        )

    frame_number = int(seconds * fps)  # floor: the frame on screen at `seconds`
    note = None

    nb_frames = probe.get("nb_frames")
    duration = probe.get("duration") or probe.get("container_duration")
    max_frame = None
    if nb_frames:
        max_frame = int(nb_frames) - 1
    elif duration:
        max_frame = max(0, int(float(duration) * fps) - 1)

    if max_frame is not None and frame_number > max_frame:
        note = (f"Computed frame {frame_number} exceeds the last frame ({max_frame}); clamped. "
                f"The timestamp may be past the end of the video stream.")
        frame_number = max_frame

    return frame_number, frame_number / fps, note


def load_probe(media_key: str) -> dict[str, Any]:
    paths.validate_media_key(media_key)
    probe_file = paths.probe_path(media_key)
    if not probe_file.exists():
        raise InvalidInputError(
            f"No probe data at {probe_file}. Run: python -m app.core.audio <video_url>"
        )
    try:
        return json.loads(probe_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrameError(f"{probe_file} is corrupt: {exc}. Delete it and re-run.") from exc


def _url_is_stale(media: ResolvedMedia) -> bool:
    """Refreshing proactively is cheaper than a doomed ffmpeg run plus the
    re-resolve anyway. The margin exists because extraction is not instant."""
    expires_at = media.video.expires_at
    if not expires_at:
        return False
    return time.time() >= (expires_at - config.FRAME_URL_EXPIRY_MARGIN_SECONDS)


def frame_filename(seconds: float) -> str:
    """Deterministic, so repeat queries reuse the same file and the directory
    sorts chronologically."""
    return f"frame_{int(round(seconds * 1000)):09d}.png"


def _run_extraction(media: ResolvedMedia, seconds: float, destination: Path) -> None:
    """One ffmpeg call: coarse keyframe seek, then decode the last few frames."""
    coarse = max(0.0, seconds - config.FRAME_PREROLL_SECONDS)
    args = [
        "-ss", f"{coarse:.3f}",              # before -i: byte-range keyframe jump
        *ffmpeg.HTTP_RECONNECT_ARGS,
        *ffmpeg.build_input_headers(media.video.http_headers),
        "-i", media.video.url,
        "-ss", f"{seconds - coarse:.3f}",    # after -i: exact seek by decoding
        "-frames:v", "1", "-an", "-y", str(destination),
    ]
    ffmpeg.run_ffmpeg(args, stage=STAGE, timeout=config.FRAME_TIMEOUT_SECONDS)


def extract_frame(
    media: ResolvedMedia,
    seconds: float,
    probe: Optional[dict[str, Any]] = None,
    *,
    force: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> FrameResult:
    """Extract the frame at `seconds` and describe it.

    Retries ONCE with a freshly resolved URL. An expired signed CDN URL (403 or
    410) is by far the most common failure and is indistinguishable from others
    at this level, so rather than parse ffmpeg's stderr we simply re-resolve and
    try again; anything else fails the second time too and both errors are
    reported together.
    """
    if seconds < 0:
        raise InvalidInputError(f"Timestamp must be non-negative, got {seconds}.")

    probe = probe if probe is not None else load_probe(media.media_key)
    frame_number, frame_pts, note = frame_number_for_timestamp(probe, seconds)
    destination = paths.ensure_frames_dir(media.media_key) / frame_filename(seconds)
    started = time.time()
    re_resolved = False

    if destination.exists() and destination.stat().st_size > 0 and not force:
        report(progress_callback, STAGE, f"cache hit: {destination.name}")
    else:
        if _url_is_stale(media):
            report(progress_callback, STAGE, "signed URL near expiry, re-resolving first")
            media = resolve(media.source_url, progress_callback=progress_callback)
            re_resolved = True

        report(progress_callback, STAGE,
               f"seeking to {seconds:.3f}s in format {media.video.format_id} "
               f"(coarse {max(0.0, seconds - config.FRAME_PREROLL_SECONDS):.3f}s + fine)")
        try:
            _run_extraction(media, seconds, destination)
        except FFmpegError as first_error:
            report(progress_callback, STAGE,
                   f"extraction failed ({str(first_error).splitlines()[0]}); "
                   f"re-resolving and retrying once")
            destination.unlink(missing_ok=True)
            try:
                media = resolve(media.source_url, progress_callback=progress_callback)
            except Quest1Error as resolve_error:
                raise FrameError(
                    f"Frame extraction failed and the URL could not be re-resolved.\n"
                    f"  original error: {first_error}\n"
                    f"  re-resolve error: {resolve_error}"
                ) from first_error
            re_resolved = True
            try:
                _run_extraction(media, seconds, destination)
            except FFmpegError as second_error:
                destination.unlink(missing_ok=True)
                raise FrameError(
                    f"Frame extraction failed at {seconds:.3f}s even after re-resolving the "
                    f"stream URL.\n  first attempt: {first_error}\n  retry: {second_error}"
                ) from second_error

    if not destination.exists() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise FrameError(
            f"ffmpeg reported success but wrote no image at {destination}. "
            f"The timestamp {seconds:.3f}s may be beyond the end of the stream."
        )

    # ffmpeg can exit 0 having written something unusable; serving a corrupt
    # image to the web UI would be a confusing failure to debug.
    with destination.open("rb") as handle:
        magic = handle.read(8)
    if magic != PNG_MAGIC:
        destination.unlink(missing_ok=True)
        raise FrameError(f"{destination} is not a valid PNG (got magic bytes {magic!r}).")

    size = destination.stat().st_size
    report(progress_callback, STAGE,
           f"wrote {destination.name} ({size:,} bytes, "
           f"frame {frame_number if frame_number is not None else 'n/a'})")
    return FrameResult(
        path=str(destination), timestamp_seconds=seconds,
        frame_number=frame_number, frame_pts=frame_pts,
        fps=probe.get("fps"), is_vfr=bool(probe.get("is_vfr")),
        width=probe.get("width"), height=probe.get("height"),
        size_bytes=size, note=note, re_resolved=re_resolved,
        elapsed_seconds=time.time() - started,
    )


def extract_from_media_key(
    media_key: str,
    seconds: float,
    *,
    source_url: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> FrameResult:
    """Extract given only a media_key, re-resolving the stream URL.

    The signed URL cannot be persisted, but the PAGE url in probe.json can, so a
    fresh stream URL is obtainable from a media_key alone. The key is verified
    to still match, which catches a mismatched --url before it writes a frame
    into the wrong video's directory.
    """
    probe = load_probe(media_key)
    url = source_url or probe.get("source_url")
    if not url:
        raise InvalidInputError(
            f"probe.json for {media_key} has no source_url. Pass --url <video_url>, "
            f"or re-run: python -m app.core.audio <video_url> --force"
        )
    media = resolve(url, progress_callback=progress_callback)
    if media.media_key != media_key:
        raise InvalidInputError(
            f"URL {url} resolves to media_key {media.media_key!r}, not {media_key!r}."
        )
    return extract_frame(media, seconds, probe, force=force, progress_callback=progress_callback)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.core.frame")
    parser.add_argument("media_key")
    parser.add_argument("seconds", type=float)
    parser.add_argument("--url", default=None, help="Page URL, if probe.json has no source_url")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = extract_from_media_key(args.media_key, args.seconds,
                                        source_url=args.url, force=args.force)
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\nimage     : {result.path}")
        print(f"timestamp : {result.timestamp_seconds:.3f}s")
        print(f"frame     : {result.frame_number if result.frame_number is not None else 'null'}")
        print(f"size      : {result.width}x{result.height}, {result.size_bytes:,} bytes")
        print(f"elapsed   : {result.elapsed_seconds:.2f}s")
        if result.note:
            print(f"note      : {result.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
