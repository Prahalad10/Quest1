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
from app.core.resolve import ResolvedMedia, resolve, retry_transient
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


def _download_audio(
    media: ResolvedMedia,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Fetch the compressed AUDIO-ONLY track using yt-dlp's chunked downloader.

    WHY NOT JUST LET FFMPEG READ THE URL (which is what this used to do)
        Measured on a 797s video, fetching the identical bytes from the
        identical URL:

            ffmpeg sequential stream          399.1s
            yt-dlp chunked ranged requests      8.7s      -- 30x faster

        YouTube throttles a single long sequential read to roughly 30 KB/s,
        while serving chunked ranged requests at ~2 MB/s. The bottleneck was
        never "remote parsing"; it was the request pattern.

    WHAT THIS DOES NOT CHANGE
        The VIDEO is still never downloaded -- frame extraction continues to
        read a few hundred KB remotely via HTTP ranges. And the audio always
        landed on disk as audio.wav regardless, so this changes how the bytes
        arrive, not whether they are stored. It also matches the original
        specification, which called for `yt-dlp -f bestaudio`.

    Downloads the EXACT format resolve.py already selected, so format choice
    stays in one place. Returns the path to the compressed file, which the
    caller transcodes and then deletes.

    USED BY: ensure_audio, when AUDIO_USE_CHUNKED_DOWNLOAD is on.
    """
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:  # pragma: no cover - environment problem
        raise AudioError(
            "yt-dlp is not installed. Run: pip install -r requirements.txt"
        ) from exc

    target_dir = paths.ensure_media_dir(media.media_key)
    outtmpl = str(target_dir / "audio_src.%(ext)s")

    # Clear COMPLETED leftovers from an earlier run so we never transcode stale
    # audio -- but deliberately keep yt-dlp's own resume state (.part and .ytdl).
    #
    # WHY: a slow or fragmented host can take many minutes for one track. ok.ru
    # serves this film's audio as ~1500 DASH fragments, and deleting the partial
    # file meant every interruption restarted a 125 MB download from zero.
    # yt-dlp validates its own .part against the .ytdl marker before resuming,
    # so keeping them is safe; the "never transcode a partial file" guarantee is
    # enforced below instead, where the finished file is selected.
    for stale in target_dir.glob("audio_src.*"):
        if stale.suffix in (".part", ".ytdl") or ".part-Frag" in stale.name:
            continue
        stale.unlink(missing_ok=True)

    def hook(status: dict[str, Any]) -> None:
        """Translate yt-dlp's progress dict into our progress contract."""
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        got = status.get("downloaded_bytes") or 0
        if total:
            percent = min(100.0, got / total * 100)
            report(
                progress_callback, STAGE_AUDIO,
                f"downloading audio {got / 1e6:.1f} / {total / 1e6:.1f} MB",
                percent=percent,
            )

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # yt-dlp writes its own progress bar to stdout, which would corrupt
        # `python -m app.cli --json`. We report progress through the hook below
        # instead, so its built-in output must be off.
        "noprogress": True,
        "consoletitle": False,
        # Pin to the format resolve.py chose, rather than re-selecting here.
        "format": media.audio.format_id,
        "outtmpl": outtmpl,
        "http_chunk_size": config.AUDIO_HTTP_CHUNK_SIZE,
        "concurrent_fragment_downloads": config.AUDIO_CONCURRENT_FRAGMENTS,
        "retries": 3,
        "progress_hooks": [hook],
    }

    report(
        progress_callback, STAGE_AUDIO,
        f"fetching audio format {media.audio.format_id} "
        f"({media.audio.ext}, {media.audio.abr or '?'}kbps) in "
        f"{config.AUDIO_HTTP_CHUNK_SIZE // (1024 * 1024)}MB chunks",
    )
    def _download() -> None:
        """One download attempt. Wrapped so a flaky host gets retried.

        WHY THIS NEEDS RETRYING AT ALL: yt-dlp re-runs the EXTRACTOR here, not
        just an HTTP GET, so a host that intermittently resets connections can
        kill the audio fetch even though resolve() already succeeded moments
        earlier. Retrying only in resolve() moved the failure one stage later
        instead of fixing it.
        """
        with YoutubeDL(options) as ydl:
            ydl.download([media.source_url])

    try:
        retry_transient(
            _download,
            description="audio download",
            stage=STAGE_AUDIO,
            progress_callback=progress_callback,
        )
    except DownloadError as exc:
        raise AudioError(
            f"Could not download the audio track for {media.source_url}: {exc}. "
            f"Set QUEST1_AUDIO_CHUNKED=0 to fall back to streaming it with ffmpeg."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface the real cause
        raise AudioError(
            f"Unexpected failure downloading audio: {type(exc).__name__}: {exc}"
        ) from exc

    # Only a FINISHED file may be transcoded. Excluding yt-dlp's scratch files
    # here is what makes keeping them above safe: a run that died mid-download
    # leaves .part/.ytdl behind, and picking one of those up would transcode
    # truncated audio and silently lose the end of the video.
    produced = [
        f for f in sorted(target_dir.glob("audio_src.*"))
        if f.suffix not in (".part", ".ytdl") and ".part-Frag" not in f.name
    ]
    if not produced:
        raise AudioError(
            "yt-dlp reported success but wrote no complete audio file. "
            "Set QUEST1_AUDIO_CHUNKED=0 to fall back to ffmpeg streaming."
        )
    return produced[0]



def _audio_track_signature(media: ResolvedMedia) -> dict[str, Any]:
    """The identity of the audio track a wav was (or would be) built from.

    Only fields that CHANGE THE SOUND belong here. The signed URL is excluded on
    purpose -- it rotates constantly, and treating a new URL for the same track
    as a different track would re-download the audio on every run.

    USED BY: _audio_cache_is_stale and ensure_audio.
    """
    return {
        "format_id": media.audio.format_id,
        "language": media.audio.language,
        "abr": media.audio.abr,
        "ext": media.audio.ext,
    }


def _audio_cache_is_stale(media: ResolvedMedia) -> Optional[str]:
    """Reason the cached wav no longer matches the track we would now pick, else None.

    WHY THIS IS NEEDED: audio.wav used to be reused whenever it existed. When
    the track selector was fixed to prefer the ORIGINAL language over the
    highest bitrate, every already-cached wav built from a dub would have been
    reused anyway, and rebuilding the index would have re-read the same wrong
    audio. Existence is not freshness.

    A wav with no sidecar is treated as stale ONLY when the source has more than
    one audio track, so single-track videos cached before this existed are not
    all re-downloaded for nothing.

    USED BY: ensure_audio.
    """
    meta_file = paths.audio_meta_path(media.media_key)
    wanted = _audio_track_signature(media)

    if not meta_file.exists():
        if getattr(media, "audio_track_count", 1) > 1:
            return "cached audio predates track selection and this video has several tracks"
        return None

    try:
        cached = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "audio.meta.json is unreadable"

    if cached.get("format_id") != wanted["format_id"]:
        return (f"cached audio came from format {cached.get('format_id')} "
                f"(lang={cached.get('language')}), now selecting {wanted['format_id']} "
                f"(lang={wanted['language']})")
    return None


def _write_audio_meta(media: ResolvedMedia) -> None:
    """Record which track the wav on disk was built from. USED BY: ensure_audio."""
    payload = _audio_track_signature(media)
    payload["written_at"] = time.time()
    meta_file = paths.audio_meta_path(media.media_key)
    temp = meta_file.with_suffix(meta_file.suffix + ".part")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, meta_file)



def _discard_derived_artifacts(media_key: str, progress_callback: Optional[ProgressCallback]) -> None:
    """Delete the transcript and index built from a wav we are about to replace.

    WHY THIS IS NOT OPTIONAL: the transcript is a pure function of the audio. If
    the audio changes and the index does not, ensure_index finds a structurally
    valid index -- right INDEX_VERSION, right schema, right word count -- and
    reuses it without complaint. That is exactly what happened after the dub
    fix: the English audio was re-fetched correctly and the Arabic transcript
    was then served from cache, so the query still missed and nothing anywhere
    reported an error.

    Invalidating the input has to invalidate everything derived from it.

    USED BY: ensure_audio, on the stale-cache path only.
    """
    for path in (paths.transcript_path(media_key), paths.index_path(media_key)):
        if path.exists():
            path.unlink()
            report(progress_callback, STAGE_AUDIO,
                   f"discarded {path.name} -- it was built from the previous audio")


def ensure_audio(
    media: ResolvedMedia,
    *,
    force: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Produce data/{media_key}/audio.wav, reusing it if already present.

    Fetches the AUDIO-ONLY track (chunked, via yt-dlp) and transcodes it to
    16kHz mono PCM. No video stream is ever opened -- the "do not download the
    video" constraint in practice.

    16kHz mono PCM because that is exactly what Whisper resamples to internally;
    producing it up front avoids a second resample at transcribe time.

    Raises AudioError on a truncated or malformed result, FFmpegError if ffmpeg
    itself fails or is missing.

    USED BY: app/service.py (stage 2). Runnable standalone via __main__.
    """
    paths.ensure_media_dir(media.media_key)
    wav = paths.audio_path(media.media_key)

    if wav.exists() and wav.stat().st_size > 0 and not force:
        stale = _audio_cache_is_stale(media)
        if stale is None:
            report(progress_callback, STAGE_AUDIO,
                   f"cache hit: {wav} ({wav.stat().st_size:,} bytes, "
                   f"lang={media.audio.language or 'n/a'})")
            return wav
        # Re-fetching is the whole point: a wav from the wrong track transcribes
        # perfectly into the wrong language and every query then misses.
        report(progress_callback, STAGE_AUDIO, f"re-fetching audio -- {stale}")
        _discard_derived_artifacts(media.media_key, progress_callback)

    if media.audio.vcodec not in (None, "none"):
        report(
            progress_callback, STAGE_AUDIO,
            f"WARNING: no audio-only format available; using progressive format "
            f"{media.audio.format_id}, which costs video bytes.",
        )

    temp = _atomic_target(wav)
    temp.unlink(missing_ok=True)
    started = time.time()

    # --- Fast path: fetch the compressed audio with chunked ranged requests --
    # 30x faster than letting ffmpeg read the URL sequentially; see
    # _download_audio for the measurements.
    if config.AUDIO_USE_CHUNKED_DOWNLOAD:
        compressed = None
        try:
            compressed = _download_audio(media, progress_callback=progress_callback)
            report(
                progress_callback, STAGE_AUDIO,
                f"transcoding {compressed.name} -> 16kHz mono wav",
            )
            ffmpeg.run_ffmpeg(
                [
                    "-i", str(compressed),
                    "-vn", "-map", "a:0",
                    "-ac", str(config.AUDIO_CHANNELS),
                    "-ar", str(config.AUDIO_SAMPLE_RATE),
                    "-c:a", "pcm_s16le", "-f", "wav",
                    "-y", str(temp),
                ],
                total_duration=media.duration,
                stage=STAGE_AUDIO,
                progress_callback=progress_callback,
            )
        except (AudioError, FFmpegError):
            temp.unlink(missing_ok=True)
            raise
        finally:
            # The compressed original is an intermediate, not an artifact.
            if compressed is not None:
                compressed.unlink(missing_ok=True)

        if not temp.exists() or temp.stat().st_size == 0:
            temp.unlink(missing_ok=True)
            raise AudioError("ffmpeg reported success but produced an empty wav file.")
        os.replace(temp, wav)
        stats = _validate_wav(wav, media.duration)
        _write_audio_meta(media)
        report(
            progress_callback, STAGE_AUDIO,
            f"wrote {wav} -- {stats['size_bytes']:,} bytes, {stats['duration']:.1f}s, "
            f"lang={media.audio.language or 'n/a'} in {time.time() - started:.1f}s",
        )
        return wav

    # --- Fallback: ffmpeg reads the remote URL directly ----------------------
    # Kept for hosts whose audio ffmpeg can read but yt-dlp cannot fetch as a
    # file. Much slower against a throttling CDN.
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
    _write_audio_meta(media)
    report(
        progress_callback, STAGE_AUDIO,
        f"wrote {wav} -- {stats['size_bytes']:,} bytes, {stats['duration']:.1f}s, "
        f"lang={media.audio.language or 'n/a'} in {time.time() - started:.1f}s",
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
