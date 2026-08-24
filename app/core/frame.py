"""Turn a timestamp into a frame number and a PNG -- without downloading the video.

Two separate jobs:

    frame_number_for_timestamp()  arithmetic over probe.json. Pure, testable,
                                  no network. Returns None for VFR sources with
                                  an explanation rather than a fabricated number.
    extract_frame()               ffmpeg reads a few hundred KB of the remote
                                  video via HTTP range requests and writes one
                                  PNG. The video is never downloaded.

Seeking uses the two-stage pattern: a coarse `-ss` BEFORE `-i` (input seeking --
ffmpeg jumps straight to the nearest keyframe using byte ranges, reading almost
nothing) and a fine `-ss` AFTER `-i` (output seeking -- decodes forward frame by
frame to land exactly). Coarse alone is fast but snaps to a keyframe; fine alone
is exact but decodes the file from the start. Together: fast and exact.

Signed CDN URLs expire. If extraction fails, we re-resolve the page URL once and
retry with a fresh stream URL.

Run directly:
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
    """The frame could not be located or extracted.

    Raised only after a re-resolve and retry have both failed, or when ffmpeg
    reported success but produced something that is not a valid PNG.
    """


@dataclass
class FrameResult:
    """Everything the CLI and the web layer need to report about one frame.

    `note` and `frame_number` work together: whenever frame_number is None, note
    explains why in a sentence meant to be shown to the user. Never present one
    without the other.

    USED BY: app/service.py, which copies these fields into DialogueResult.
    """

    path: str
    timestamp_seconds: float
    frame_number: Optional[int]   # None for VFR -- see `note`
    frame_pts: Optional[float]    # presentation time of that frame, seconds
    fps: Optional[float]
    is_vfr: bool
    width: Optional[int]
    height: Optional[int]
    size_bytes: int
    note: Optional[str]           # why frame_number is None, or a clamp warning
    re_resolved: bool             # True if the signed URL had to be refreshed
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """JSON form. USED BY: `python -m app.core.frame --json`."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Frame arithmetic (pure -- no network, no ffmpeg)
# --------------------------------------------------------------------------- #

def frame_number_for_timestamp(
    probe: dict[str, Any],
    seconds: float,
) -> tuple[Optional[int], Optional[float], Optional[str]]:
    """Return (frame_number, frame_pts, note) for `seconds` given probe data.

    PURE ARITHMETIC -- no network, no ffmpeg -- which is why it can be unit
    tested against synthetic probe dicts covering CFR, VFR, unknown fps and
    past-the-end timestamps.

    For a constant-frame-rate video, frame N covers [N/fps, (N+1)/fps), so the
    frame visible at time t is floor(t * fps). Floor, not round: the frame ON
    SCREEN at t is the one whose interval contains t.

    For VFR the mapping is not a multiplication at all -- frame durations vary
    across the file -- so any number computed this way would be wrong. Returns
    None plus an explanatory note instead of guessing, because a confidently
    wrong frame number is worse than an honest absence. The extracted IMAGE is
    still correct either way; only the integer is unavailable.

    USED BY: extract_frame, and directly by tests.
    """
    if seconds < 0:
        raise InvalidInputError(f"Timestamp must be non-negative, got {seconds}.")

    if probe.get("is_vfr"):
        return None, None, (
            "Source is variable-frame-rate (r_frame_rate "
            f"{probe.get('r_frame_rate')} != avg_frame_rate {probe.get('avg_frame_rate')}), "
            "so a frame number cannot be derived from the timestamp by arithmetic. "
            "The extracted image is still the correct frame for this timestamp."
        )

    fps = probe.get("fps")
    if not fps or fps <= 0:
        return None, None, (
            "Frame rate is unknown for this stream, so no frame number can be "
            "computed. The extracted image is still correct for this timestamp."
        )

    frame_number = int(seconds * fps)  # floor: the frame on screen at `seconds`
    note = None

    duration = probe.get("duration") or probe.get("container_duration")
    nb_frames = probe.get("nb_frames")
    max_frame = None
    if nb_frames:
        max_frame = int(nb_frames) - 1
    elif duration:
        max_frame = max(0, int(float(duration) * fps) - 1)

    if max_frame is not None and frame_number > max_frame:
        note = (
            f"Computed frame {frame_number} exceeds the last frame ({max_frame}); "
            f"clamped. The timestamp may be past the end of the video stream."
        )
        frame_number = max_frame

    return frame_number, frame_number / fps, note


# --------------------------------------------------------------------------- #
# Probe / media loading
# --------------------------------------------------------------------------- #

def load_probe(media_key: str) -> dict[str, Any]:
    """Read probe.json for a media_key.

    Distinguishes "not produced yet" (InvalidInputError naming the command to
    run) from "corrupt" (FrameError), because the fixes differ.

    USED BY: extract_frame when no probe was passed in, and
    extract_from_media_key.
    """
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
    """True when the signed video URL is at or near its expiry.

    WHY CHECK IN ADVANCE: re-resolving proactively costs one yt-dlp call, while
    discovering the expiry through a failed extraction costs a doomed ffmpeg
    invocation AND the re-resolve anyway. The margin exists because extraction
    is not instantaneous -- a URL valid for two more seconds is not useful.

    USED BY: extract_frame, before its first attempt.
    """
    expires_at = media.video.expires_at
    if not expires_at:
        return False
    return time.time() >= (expires_at - config.FRAME_URL_EXPIRY_MARGIN_SECONDS)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def frame_filename(seconds: float) -> str:
    """Deterministic per-timestamp filename, so repeat queries reuse the same file.

    Millisecond precision and zero padding mean the same timestamp always maps to
    the same filename, and the directory sorts chronologically.

    USED BY: extract_frame (writes) and app/service.py (checks existence to
    report whether the frame stage was cached).
    """
    return f"frame_{int(round(seconds * 1000)):09d}.png"


def _run_extraction(media: ResolvedMedia, seconds: float, destination: Path) -> None:
    """One ffmpeg invocation: coarse seek BEFORE -i, fine seek AFTER -i.

    THE TWO-STAGE SEEK, which is what makes single-frame extraction cheap:

      -ss before -i   INPUT seeking. ffmpeg uses byte ranges to jump straight to
                      the nearest keyframe, reading almost nothing. Fast, but it
                      lands on a keyframe rather than the exact frame.
      -ss after -i    OUTPUT seeking. Decodes forward frame by frame to land
                      exactly. Accurate, but would decode from the start of the
                      file if used alone.

    Together: jump to just before the target, then decode the last few frames.
    A frame from 8 minutes into a 1080p video costs a few hundred KB and a few
    seconds instead of a 100MB download.

    USED BY: extract_frame, which calls it once and possibly a second time after
    re-resolving.
    """
    coarse = max(0.0, seconds - config.FRAME_PREROLL_SECONDS)
    fine = seconds - coarse

    args = [
        # --- input options ---
        "-ss", f"{coarse:.3f}",          # BEFORE -i: fast keyframe seek via byte ranges
        *ffmpeg.HTTP_RECONNECT_ARGS,
        *ffmpeg.build_input_headers(media.video.http_headers),
        "-i", media.video.url,
        # --- output options ---
        "-ss", f"{fine:.3f}",            # AFTER -i: exact seek by decoding forward
        "-frames:v", "1",
        "-an",
        "-y", str(destination),
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
    """Extract the frame at `seconds` to data/{media_key}/frames/ and describe it.

    Retries ONCE with a freshly resolved URL if the first attempt fails. A 403 or
    410 from an expired signed CDN URL is by far the most common cause, and it is
    indistinguishable from other failures at this level -- so rather than parse
    ffmpeg stderr for status codes, we simply re-resolve and try again. Anything
    that is not an expiry will fail the second time too, and both errors are
    reported together.

    The PNG magic-byte check at the end guards the case where ffmpeg exits 0 but
    writes something unusable; serving a corrupt image to the web UI would be a
    confusing failure to debug.

    USED BY: app/service.py (stage 6) and extract_from_media_key.
    """
    if seconds < 0:
        raise InvalidInputError(f"Timestamp must be non-negative, got {seconds}.")

    probe = probe if probe is not None else load_probe(media.media_key)
    frame_number, frame_pts, note = frame_number_for_timestamp(probe, seconds)

    frames_dir = paths.ensure_frames_dir(media.media_key)
    destination = frames_dir / frame_filename(seconds)
    started = time.time()
    re_resolved = False

    if destination.exists() and destination.stat().st_size > 0 and not force:
        report(progress_callback, STAGE, f"cache hit: {destination.name}")
    else:
        # Refresh proactively when the URL is known to be near expiry -- cheaper
        # than letting ffmpeg fail and retrying.
        if _url_is_stale(media):
            report(progress_callback, STAGE, "signed URL near expiry, re-resolving first")
            media = resolve(media.source_url, check_ranges=False,
                            progress_callback=progress_callback)
            re_resolved = True

        report(
            progress_callback, STAGE,
            f"seeking to {seconds:.3f}s in format {media.video.format_id} "
            f"(coarse {max(0.0, seconds - config.FRAME_PREROLL_SECONDS):.3f}s + fine)",
        )
        try:
            _run_extraction(media, seconds, destination)
        except FFmpegError as first_error:
            # The overwhelmingly common cause is an expired signed URL (403/410).
            # Re-resolve once and retry; anything else fails on the second try too.
            report(
                progress_callback, STAGE,
                f"extraction failed ({str(first_error).splitlines()[0]}); "
                f"re-resolving {media.source_url} and retrying once",
            )
            destination.unlink(missing_ok=True)
            try:
                media = resolve(media.source_url, check_ranges=False,
                                progress_callback=progress_callback)
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
                    f"Frame extraction failed at {seconds:.3f}s even after re-resolving "
                    f"the stream URL.\n  first attempt: {first_error}\n"
                    f"  retry: {second_error}"
                ) from second_error

    if not destination.exists() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise FrameError(
            f"ffmpeg reported success but wrote no image at {destination}. "
            f"The timestamp {seconds:.3f}s may be beyond the end of the stream."
        )

    with destination.open("rb") as handle:
        magic = handle.read(8)
    if magic != PNG_MAGIC:
        destination.unlink(missing_ok=True)
        raise FrameError(f"{destination} is not a valid PNG (got magic bytes {magic!r}).")

    size = destination.stat().st_size
    report(
        progress_callback, STAGE,
        f"wrote {destination.name} ({size:,} bytes, frame "
        f"{frame_number if frame_number is not None else 'n/a'})",
    )
    return FrameResult(
        path=str(destination),
        timestamp_seconds=seconds,
        frame_number=frame_number,
        frame_pts=frame_pts,
        fps=probe.get("fps"),
        is_vfr=bool(probe.get("is_vfr")),
        width=probe.get("width"),
        height=probe.get("height"),
        size_bytes=size,
        note=note,
        re_resolved=re_resolved,
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
    """Extract a frame given only a media_key, re-resolving the stream URL.

    WHY IT EXISTS: satisfies the rule that every module is independently runnable.
    The signed stream URL cannot be persisted because it expires, but the PAGE
    url stored in probe.json can be, so a fresh stream URL is obtainable from a
    media_key alone.

    Verifies that the URL still resolves to the SAME media_key, which catches a
    mismatched --url argument before it writes a frame into the wrong video
    directory.

    USED BY: the __main__ block. app/service.py calls extract_frame directly
    because it already holds a resolved media object.
    """
    probe = load_probe(media_key)
    url = source_url or probe.get("source_url")
    if not url:
        raise InvalidInputError(
            f"probe.json for {media_key} has no source_url (it predates this field). "
            f"Pass --url <video_url>, or re-run: python -m app.core.audio <video_url> --force"
        )

    media = resolve(url, check_ranges=False, progress_callback=progress_callback)
    if media.media_key != media_key:
        raise InvalidInputError(
            f"URL {url} resolves to media_key {media.media_key!r}, not {media_key!r}."
        )
    return extract_frame(
        media, seconds, probe, force=force, progress_callback=progress_callback,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry point: extract the frame at a timestamp and report it.

    WHY: lets frame extraction and the timestamp->frame arithmetic be checked
    against any timestamp without running ASR or matching first.

    USED BY: `python -m app.core.frame <media_key> <seconds>`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.core.frame",
        description="Extract the video frame at a timestamp via ranged HTTP seek.",
    )
    parser.add_argument("media_key", help="media_key (see app.core.resolve)")
    parser.add_argument("seconds", type=float, help="Timestamp in seconds, e.g. 4.5")
    parser.add_argument("--url", default=None,
                        help="Page URL, if probe.json has no source_url")
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)

    try:
        result = extract_from_media_key(
            args.media_key, args.seconds, source_url=args.url, force=args.force,
        )
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print()
    print(f"timestamp : {result.timestamp_seconds:.3f}s")
    print(f"frame     : {result.frame_number if result.frame_number is not None else 'null (VFR)'}")
    print(f"frame_pts : {f'{result.frame_pts:.4f}s' if result.frame_pts is not None else 'n/a'}")
    print(f"fps       : {result.fps}  (VFR: {result.is_vfr})")
    print(f"size      : {result.width}x{result.height}")
    print(f"image     : {result.path}")
    print(f"bytes     : {result.size_bytes:,}")
    print(f"elapsed   : {result.elapsed_seconds:.2f}s")
    print(f"re-resolved: {result.re_resolved}")
    if result.note:
        print()
        print(f"note      : {result.note}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
