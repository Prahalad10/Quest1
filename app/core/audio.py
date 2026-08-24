"""Fetch the audio track and probe the video stream -- without downloading video.

Two independent, cacheable artifacts under data/{media_key}/:

    audio.wav   ffmpeg streams the audio-ONLY format URL and transcodes it to
                16kHz mono PCM. No video bytes are ever read.
    probe.json  ffprobe reads only the header of the VIDEO format URL to learn
                fps / dimensions / VFR. Step 5 needs this to turn a timestamp
                into a frame number.

Both are written to a .part file and atomically renamed, so a file that exists
is always complete -- which is what makes "skip if present" safe rather than a
silent-corruption trap.

Run directly:
    python -m app.core.audio "<video_url>"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from app import config, paths
from app.core import ffmpeg
from app.core.resolve import ResolvedMedia, resolve
from app.errors import AudioError, FFmpegError, Quest1Error
from app.progress import ProgressCallback, report

STAGE_AUDIO = "audio"
STAGE_PROBE = "probe"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def parse_frame_rate(value: Optional[str]) -> Optional[float]:
    """Parse an ffprobe frame-rate string into a float.

    ffprobe reports rates as exact rationals ('30000/1001' for NTSC 29.97) so no
    precision is lost in its own output. We need a float to do timestamp->frame
    arithmetic, but the ORIGINAL string is persisted alongside it so the exact
    rational is never thrown away.

    Returns None for absent, malformed, or zero-denominator values -- '0/0' is
    what ffprobe emits when it genuinely does not know, and treating that as 0.0
    would produce a divide-by-zero later.

    USED BY: ensure_probe (both r_frame_rate and avg_frame_rate).
    """
    if not value or not isinstance(value, str):
        return None
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            numerator, denominator = float(num), float(den)
        except ValueError:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    try:
        return float(value)
    except ValueError:
        return None


def _atomic_target(final: Path) -> Path:
    """Temporary sibling path used while a file is still being written.

    WHY ATOMIC WRITES MATTER HERE: every artifact is cached with a "skip if the
    file exists" check. If an interrupted run left a half-written audio.wav in
    place, every later run would reuse the truncated file and silently lose
    dialogue. Writing to .part and renaming means a file that EXISTS is always
    COMPLETE, which is what makes skip-if-present safe.

    USED BY: ensure_audio and _write_json_atomic.
    """
    return final.with_suffix(final.suffix + ".part")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a .part file then rename, so readers never see a partial doc.

    USED BY: ensure_probe. index.py has its own copy for the same reason.
    """
    temp = _atomic_target(path)
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #

def _validate_wav(wav: Path, expected_duration: float) -> dict[str, Any]:
    """Confirm the produced wav is real, correctly formatted, and not truncated.

    WHY THIS EXISTS: ffmpeg can exit 0 having produced a file that is unusable --
    an empty stream, the wrong sample rate, or a stream cut short when the
    connection dropped. A truncated wav is the dangerous case: ASR would happily
    transcribe the first half of the video and the pipeline would then report
    "not found" for any line spoken in the second half, with no indication that
    anything went wrong.

    Fails loudly rather than returning a flag, because there is no sensible way
    to continue with bad audio.

    USED BY: ensure_audio, immediately after the transcode.
    """
    info = ffmpeg.probe_media(str(wav), select_streams="a:0")
    streams = info.get("streams") or []
    if not streams:
        raise AudioError(f"{wav} contains no audio stream -- the transcode produced silence.")

    stream = streams[0]
    sample_rate = int(stream.get("sample_rate") or 0)
    channels = int(stream.get("channels") or 0)
    if sample_rate != config.AUDIO_SAMPLE_RATE or channels != config.AUDIO_CHANNELS:
        raise AudioError(
            f"{wav} is {sample_rate}Hz/{channels}ch, expected "
            f"{config.AUDIO_SAMPLE_RATE}Hz/{config.AUDIO_CHANNELS}ch. Delete it and re-run."
        )

    duration = float((info.get("format") or {}).get("duration") or stream.get("duration") or 0.0)
    tolerance = max(
        float(config.AUDIO_DURATION_TOLERANCE_SECONDS),
        expected_duration * config.AUDIO_DURATION_TOLERANCE_RATIO,
    )
    if expected_duration - duration > tolerance:
        raise AudioError(
            f"{wav} is {duration:.1f}s but the video is {expected_duration:.1f}s -- the audio "
            f"fetch was truncated (tolerance {tolerance:.1f}s). Delete it and re-run."
        )
    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "size_bytes": wav.stat().st_size,
    }


def ensure_audio(
    media: ResolvedMedia,
    *,
    force: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Produce data/{media_key}/audio.wav, reusing it if already present.

    ffmpeg streams the AUDIO-ONLY format URL and transcodes on the fly -- the
    compressed audio is never saved, and no video stream is ever opened. This is
    the "do not download the video" constraint in practice.

    16kHz mono PCM because that is exactly what Whisper resamples to internally;
    producing it up front avoids a second resample at transcribe time.

    Raises AudioError on a truncated or malformed result, FFmpegError if ffmpeg
    itself fails or is missing.

    USED BY: app/service.py (stage 2). Runnable standalone via __main__.
    """
    paths.ensure_media_dir(media.media_key)
    wav = paths.audio_path(media.media_key)

    if wav.exists() and wav.stat().st_size > 0 and not force:
        report(progress_callback, STAGE_AUDIO, f"cache hit: {wav} ({wav.stat().st_size:,} bytes)")
        return wav

    if media.audio.vcodec not in (None, "none"):
        report(
            progress_callback, STAGE_AUDIO,
            f"WARNING: no audio-only format available; using progressive format "
            f"{media.audio.format_id}, which costs video bytes.",
        )

    temp = _atomic_target(wav)
    temp.unlink(missing_ok=True)
    report(
        progress_callback, STAGE_AUDIO,
        f"streaming audio format {media.audio.format_id} "
        f"({media.audio.ext}, {media.audio.abr or '?'}kbps) -> 16kHz mono wav",
    )

    args = [
        *ffmpeg.HTTP_RECONNECT_ARGS,
        *ffmpeg.build_input_headers(media.audio.http_headers),
        "-i", media.audio.url,
        "-vn",                                      # never decode a video stream
        "-map", "a:0",
        "-ac", str(config.AUDIO_CHANNELS),
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "-y", str(temp),
    ]
    started = time.time()
    try:
        ffmpeg.run_ffmpeg(
            args,
            total_duration=media.duration,
            stage=STAGE_AUDIO,
            progress_callback=progress_callback,
        )
    except FFmpegError:
        temp.unlink(missing_ok=True)
        raise

    if not temp.exists() or temp.stat().st_size == 0:
        temp.unlink(missing_ok=True)
        raise AudioError("ffmpeg reported success but produced an empty wav file.")

    os.replace(temp, wav)
    stats = _validate_wav(wav, media.duration)
    report(
        progress_callback, STAGE_AUDIO,
        f"wrote {wav} -- {stats['size_bytes']:,} bytes, {stats['duration']:.1f}s "
        f"in {time.time() - started:.1f}s",
    )
    return wav


# --------------------------------------------------------------------------- #
# Video probe
# --------------------------------------------------------------------------- #

def ensure_probe(
    media: ResolvedMedia,
    *,
    force: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Produce data/{media_key}/probe.json from the VIDEO stream header.

    Reads only the container header via HTTP range requests -- ffprobe does not
    download the video. For a 10-minute 1080p file this takes about a second.

    Records r_frame_rate and avg_frame_rate SEPARATELY: when they disagree the
    source is variable-frame-rate, and timestamp x fps is then simply the wrong
    formula. frame.py reads `is_vfr` and returns a null frame number with an
    explanation rather than a plausible-looking wrong integer.

    Also persists `source_url` so frame.py can re-resolve a fresh signed URL
    from a media_key alone.

    USED BY: app/service.py (stage 3) and frame.py (via load_probe).
    """
    paths.ensure_media_dir(media.media_key)
    probe_file = paths.probe_path(media.media_key)

    if probe_file.exists() and probe_file.stat().st_size > 0 and not force:
        try:
            cached = json.loads(probe_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AudioError(
                f"{probe_file} is corrupt ({exc}). Delete it and re-run."
            ) from exc
        report(progress_callback, STAGE_PROBE, f"cache hit: {probe_file}")
        return cached

    report(progress_callback, STAGE_PROBE, f"probing video format {media.video.format_id}")
    info = ffmpeg.probe_media(
        media.video.url,
        http_headers=media.video.http_headers,
        select_streams="v:0",
    )
    streams = info.get("streams") or []
    if not streams:
        raise AudioError(
            f"ffprobe found no video stream in format {media.video.format_id}. "
            "Frame extraction would be impossible."
        )

    stream = streams[0]
    container = info.get("format") or {}

    r_rate_raw = stream.get("r_frame_rate")
    avg_rate_raw = stream.get("avg_frame_rate")
    r_rate = parse_frame_rate(r_rate_raw)
    avg_rate = parse_frame_rate(avg_rate_raw)

    # Disagreement between the two means the container does not hold a constant
    # frame rate. Tiny float noise is not VFR, so compare relatively.
    is_vfr = False
    if r_rate and avg_rate:
        is_vfr = abs(r_rate - avg_rate) / max(r_rate, avg_rate) > 0.01
    elif not avg_rate and not r_rate:
        is_vfr = True  # unknown rate is as unusable as a varying one

    fps = avg_rate or r_rate

    def _as_float(value: Any) -> Optional[float]:
        """Coerce an ffprobe field to float, or None when absent/unparseable.

        ffprobe omits duration entirely for some remote video-only streams and
        emits the string 'N/A' for others, so this must tolerate both.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    probe = {
        "media_key": media.media_key,
        "probed_at": time.time(),
        "source_format_id": media.video.format_id,
        # Stored so frame.py can re-resolve from a media_key alone: the stream
        # URL itself is signed and expires, but the page URL does not.
        "source_url": media.source_url,
        "codec_name": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "r_frame_rate": r_rate_raw,
        "r_frame_rate_value": r_rate,
        "avg_frame_rate": avg_rate_raw,
        "avg_frame_rate_value": avg_rate,
        "fps": fps,
        "is_vfr": is_vfr,
        "nb_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "stream_duration": _as_float(stream.get("duration")),
        "container_duration": _as_float(container.get("duration")),
        # The metadata duration is the most trustworthy figure for a remote
        # video-only stream, whose container header often omits it.
        "duration": _as_float(container.get("duration")) or media.duration,
        "metadata_duration": media.duration,
    }
    _write_json_atomic(probe_file, probe)

    note = "VARIABLE frame rate" if is_vfr else f"{fps:.3f} fps" if fps else "unknown fps"
    report(
        progress_callback, STAGE_PROBE,
        f"wrote {probe_file} -- {probe['width']}x{probe['height']}, {note}",
    )
    return probe


def prepare_media(
    media: ResolvedMedia,
    *,
    force: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[Path, dict[str, Any]]:
    """Convenience: produce both cached artifacts for a resolved video.

    USED BY: callers that want the full Step 2 output in one call. app/service.py
    calls ensure_audio and ensure_probe separately so it can report which of the
    two was individually cached.
    """
    wav = ensure_audio(media, force=force, progress_callback=progress_callback)
    probe = ensure_probe(media, force=force, progress_callback=progress_callback)
    return wav, probe


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry point: fetch audio and/or probe a video, then report.

    WHY: makes Step 2 checkable on its own. The printed size-vs-expected ratio is
    the quickest way to confirm the wav is intact -- 16kHz mono 16-bit PCM is
    exactly 32000 bytes per second, so the ratio should sit very close to 1.000.

    USED BY: `python -m app.core.audio <video_url> [--probe-only|--audio-only]`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.core.audio",
        description="Fetch audio-only -> 16kHz mono wav, and probe the video stream.",
    )
    parser.add_argument("url", help="Video URL")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--audio-only", action="store_true", help="Skip the video probe")
    parser.add_argument("--probe-only", action="store_true", help="Skip the audio fetch")
    args = parser.parse_args(argv)

    try:
        media = resolve(args.url, check_ranges=False)
        wav = None if args.probe_only else ensure_audio(media, force=args.force)
        probe = None if args.audio_only else ensure_probe(media, force=args.force)
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"media_key : {media.media_key}")
    print(f"title     : {media.title}")
    print(f"duration  : {media.duration:.2f}s")

    if wav is not None:
        size = wav.stat().st_size
        expected = media.duration * config.AUDIO_BYTES_PER_SECOND
        print()
        print(f"audio.wav : {wav}")
        print(f"  size    : {size:,} bytes ({size / 1_048_576:.2f} MiB)")
        print(f"  expected: ~{expected:,.0f} bytes at "
              f"{config.AUDIO_SAMPLE_RATE}Hz/{config.AUDIO_CHANNELS}ch/16-bit")
        print(f"  ratio   : {size / expected:.3f} of expected")

    if probe is not None:
        print()
        print(f"probe.json: {paths.probe_path(media.media_key)}")
        print(f"  size    : {probe['width']}x{probe['height']} ({probe['codec_name']})")
        print(f"  r_frame_rate  : {probe['r_frame_rate']} -> {probe['r_frame_rate_value']}")
        print(f"  avg_frame_rate: {probe['avg_frame_rate']} -> {probe['avg_frame_rate_value']}")
        print(f"  fps     : {probe['fps']}")
        print(f"  VFR     : {probe['is_vfr']}"
              + ("  <-- frame number will be reported as null in Step 5" if probe["is_vfr"] else ""))
        print(f"  duration: {probe['duration']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
