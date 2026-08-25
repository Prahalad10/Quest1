"""Fetch the audio track as a 16kHz mono wav, and probe the video stream.

Two artifacts: audio.wav (the ASR input) and probe.json (fps, dimensions, VFR
flag, source_url). Both are written to a .part file and atomically renamed, so
a file that exists is always complete -- which is what makes "skip if present"
safe rather than a silent-corruption trap.

The audio is fetched with yt-dlp's chunked ranged downloader rather than by
letting ffmpeg read the URL sequentially: 399s -> 8.7s on a 797s video, because
YouTube throttles one long read. This does not download the video -- only the
audio track. The exception is a host offering no audio-only track, where the
smallest progressive stream is used and a warning is emitted.

    python -m app.core.audio <url> [--audio-only|--probe-only]
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


def parse_frame_rate(value: Optional[str]) -> Optional[float]:
    """ffprobe reports rates as "30000/1001"; return it as a float."""
    if not value:
        return None
    text = str(value).strip()
    try:
        if "/" in text:
            num, _, den = text.partition("/")
            denominator = float(den)
            return float(num) / denominator if denominator else None
        return float(text) or None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _atomic_target(final: Path) -> Path:
    return final.with_suffix(final.suffix + ".part")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = _atomic_target(path)
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def _validate_wav(wav: Path, expected_duration: float) -> dict[str, Any]:
    """Confirm the wav is real, correctly formatted, and not truncated.

    ffmpeg can exit 0 having produced an unusable file. Truncation is the
    dangerous case: ASR would transcribe the first half and the pipeline would
    report "not found" for every line in the second, with nothing to show that
    anything went wrong.
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
    tolerance = max(float(config.AUDIO_DURATION_TOLERANCE_SECONDS),
                    expected_duration * config.AUDIO_DURATION_TOLERANCE_RATIO)
    if expected_duration - duration > tolerance:
        raise AudioError(
            f"{wav} is {duration:.1f}s but the video is {expected_duration:.1f}s -- the audio "
            f"fetch was truncated (tolerance {tolerance:.1f}s). Delete it and re-run."
        )
    return {"duration": duration, "sample_rate": sample_rate,
            "channels": channels, "size_bytes": wav.stat().st_size}


def _download_audio(
    media: ResolvedMedia, *, progress_callback: Optional[ProgressCallback] = None
) -> Path:
    """Fetch the compressed track with yt-dlp's chunked downloader.

    Downloads the exact format resolve.py chose, so format selection stays in
    one place.
    """
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:  # pragma: no cover - environment problem
        raise AudioError("yt-dlp is not installed. Run: pip install -r requirements.txt") from exc

    target_dir = paths.ensure_media_dir(media.media_key)

    # Clear COMPLETED leftovers but keep yt-dlp's resume state: ok.ru serves
    # some audio as ~1500 DASH fragments, and deleting the partial restarted a
    # 125 MB fetch from zero on every interruption. The "never transcode a
    # partial file" guarantee is enforced at selection below instead.
    for stale in target_dir.glob("audio_src.*"):
        if stale.suffix in (".part", ".ytdl") or ".part-Frag" in stale.name:
            continue
        stale.unlink(missing_ok=True)

    def hook(status: dict[str, Any]) -> None:
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        got = status.get("downloaded_bytes") or 0
        if total:
            report(progress_callback, STAGE_AUDIO,
                   f"downloading audio {got / 1e6:.1f} / {total / 1e6:.1f} MB",
                   percent=min(100.0, got / total * 100))

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # yt-dlp's own progress bar goes to stdout and would corrupt --json.
        "noprogress": True,
        "consoletitle": False,
        "format": media.audio.format_id,
        "outtmpl": str(target_dir / "audio_src.%(ext)s"),
        "http_chunk_size": config.AUDIO_HTTP_CHUNK_SIZE,
        "concurrent_fragment_downloads": config.AUDIO_CONCURRENT_FRAGMENTS,
        "retries": config.RESOLVE_HTTP_RETRIES,
        "progress_hooks": [hook],
    }

    report(progress_callback, STAGE_AUDIO,
           f"fetching audio format {media.audio.format_id} "
           f"({media.audio.ext}, {media.audio.abr or '?'}kbps)")

    def download() -> None:
        with YoutubeDL(options) as ydl:
            ydl.download([media.source_url])

    try:
        # yt-dlp re-runs the EXTRACTOR here, so a host that intermittently
        # resets connections can kill this even after resolve() just succeeded.
        retry_transient(download, description="audio download", stage=STAGE_AUDIO,
                        progress_callback=progress_callback)
    except DownloadError as exc:
        raise AudioError(
            f"Could not download the audio track for {media.source_url}: {exc}. "
            f"Set QUEST1_AUDIO_CHUNKED=0 to fall back to streaming it with ffmpeg."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface the real cause
        raise AudioError(
            f"Unexpected failure downloading audio: {type(exc).__name__}: {exc}"
        ) from exc

    # Only a FINISHED file may be transcoded -- this is what makes keeping the
    # resume scratch files above safe.
    produced = [f for f in sorted(target_dir.glob("audio_src.*"))
                if f.suffix not in (".part", ".ytdl") and ".part-Frag" not in f.name]
    if not produced:
        raise AudioError(
            "yt-dlp reported success but wrote no complete audio file. "
            "Set QUEST1_AUDIO_CHUNKED=0 to fall back to ffmpeg streaming."
        )
    return produced[0]


def _audio_track_signature(media: ResolvedMedia) -> dict[str, Any]:
    """Identity of the track a wav was built from.

    The signed URL is excluded deliberately: it rotates, and treating a new URL
    for the same track as a different track would re-download every run.
    """
    return {"format_id": media.audio.format_id, "language": media.audio.language,
            "abr": media.audio.abr, "ext": media.audio.ext}


def _audio_cache_is_stale(media: ResolvedMedia) -> Optional[str]:
    """Why the cached wav no longer matches the track we would now pick, else None.

    Existence is not freshness. When track selection was fixed to prefer the
    original language, every wav already built from a dub would otherwise have
    been reused, and the index would rebuild from the same wrong audio.

    A missing sidecar counts as stale only when the source has several audio
    languages, so single-language videos cached before this existed are not all
    re-downloaded for nothing.
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
    payload = _audio_track_signature(media)
    payload["written_at"] = time.time()
    _write_json_atomic(paths.audio_meta_path(media.media_key), payload)


def _discard_derived_artifacts(
    media_key: str, progress_callback: Optional[ProgressCallback]
) -> None:
    """Delete the transcript built from a wav we are about to replace.

    The transcript is a pure function of the audio. If the audio changes and
    the index does not, ensure_index finds a structurally valid index and
    reuses it -- which is exactly how corrected English audio ended up serving
    an Arabic transcript from cache, with no error anywhere.
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
    """Produce data/{media_key}/audio.wav, reusing it when it matches the track."""
    paths.ensure_media_dir(media.media_key)
    wav = paths.audio_path(media.media_key)

    if wav.exists() and wav.stat().st_size > 0 and not force:
        stale = _audio_cache_is_stale(media)
        if stale is None:
            report(progress_callback, STAGE_AUDIO,
                   f"cache hit: {wav} ({wav.stat().st_size:,} bytes, "
                   f"lang={media.audio.language or 'n/a'})")
            return wav
        report(progress_callback, STAGE_AUDIO, f"re-fetching audio -- {stale}")
        _discard_derived_artifacts(media.media_key, progress_callback)

    # is_audio_only is decided at selection time. Testing vcodec here instead
    # missed hosts that report no codec info, and stayed silent while the
    # pipeline downloaded a whole video file.
    if not media.audio.is_audio_only:
        report(progress_callback, STAGE_AUDIO,
               f"WARNING: this host offers no audio-only track; using progressive format "
               f"{media.audio.format_id}, which costs video bytes. This is the one path "
               f"that fetches video data.")

    temp = _atomic_target(wav)
    temp.unlink(missing_ok=True)
    started = time.time()
    transcode = ["-vn", "-map", "a:0",
                 "-ac", str(config.AUDIO_CHANNELS),
                 "-ar", str(config.AUDIO_SAMPLE_RATE),
                 "-c:a", "pcm_s16le", "-f", "wav", "-y", str(temp)]

    if config.AUDIO_USE_CHUNKED_DOWNLOAD:
        compressed = None
        try:
            compressed = _download_audio(media, progress_callback=progress_callback)
            report(progress_callback, STAGE_AUDIO, f"transcoding {compressed.name} -> wav")
            ffmpeg.run_ffmpeg(["-i", str(compressed), *transcode],
                              total_duration=media.duration, stage=STAGE_AUDIO,
                              progress_callback=progress_callback)
        except (AudioError, FFmpegError):
            temp.unlink(missing_ok=True)
            raise
        finally:
            if compressed is not None:
                compressed.unlink(missing_ok=True)
    else:
        # Fallback for hosts whose audio ffmpeg can read but yt-dlp cannot
        # fetch as a file. Much slower against a throttling CDN.
        report(progress_callback, STAGE_AUDIO,
               f"streaming audio format {media.audio.format_id} -> wav")
        try:
            ffmpeg.run_ffmpeg([*ffmpeg.HTTP_RECONNECT_ARGS,
                               *ffmpeg.build_input_headers(media.audio.http_headers),
                               "-i", media.audio.url, *transcode],
                              total_duration=media.duration, stage=STAGE_AUDIO,
                              progress_callback=progress_callback)
        except FFmpegError:
            temp.unlink(missing_ok=True)
            raise

    if not temp.exists() or temp.stat().st_size == 0:
        temp.unlink(missing_ok=True)
        raise AudioError("ffmpeg reported success but produced an empty wav file.")

    os.replace(temp, wav)
    stats = _validate_wav(wav, media.duration)
    _write_audio_meta(media)
    report(progress_callback, STAGE_AUDIO,
           f"wrote {wav} -- {stats['size_bytes']:,} bytes, {stats['duration']:.1f}s, "
           f"lang={media.audio.language or 'n/a'} in {time.time() - started:.1f}s")
    return wav


def ensure_probe(
    media: ResolvedMedia,
    *,
    force: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Produce probe.json from the video stream header.

    Reads only the container header via range requests -- no video is
    downloaded. r_frame_rate and avg_frame_rate are stored separately because
    when they disagree the source is variable-frame-rate and timestamp x fps is
    the wrong formula; frame.py then returns a null frame number rather than a
    plausible-looking wrong integer.
    """
    paths.ensure_media_dir(media.media_key)
    probe_file = paths.probe_path(media.media_key)

    if probe_file.exists() and probe_file.stat().st_size > 0 and not force:
        try:
            cached = json.loads(probe_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AudioError(f"{probe_file} is corrupt ({exc}). Delete it and re-run.") from exc
        report(progress_callback, STAGE_PROBE, f"cache hit: {probe_file}")
        return cached

    report(progress_callback, STAGE_PROBE, f"probing video format {media.video.format_id}")
    info = ffmpeg.probe_media(media.video.url, http_headers=media.video.http_headers,
                              select_streams="v:0")
    streams = info.get("streams") or []
    if not streams:
        raise AudioError(
            f"ffprobe found no video stream in format {media.video.format_id}. "
            "Frame extraction would be impossible."
        )

    stream, container = streams[0], (info.get("format") or {})
    r_rate = parse_frame_rate(stream.get("r_frame_rate"))
    avg_rate = parse_frame_rate(stream.get("avg_frame_rate"))

    # Float noise is not VFR, so compare relatively. An unknown rate is as
    # unusable as a varying one.
    if r_rate and avg_rate:
        is_vfr = abs(r_rate - avg_rate) / max(r_rate, avg_rate) > 0.01
    else:
        is_vfr = not (avg_rate or r_rate)
    fps = avg_rate or r_rate

    def as_float(value: Any) -> Optional[float]:
        """ffprobe omits duration for some remote streams and emits 'N/A' for others."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    probe = {
        "media_key": media.media_key,
        "probed_at": time.time(),
        "source_format_id": media.video.format_id,
        # Stored so frame.py can re-resolve from a media_key alone: the stream
        # URL is signed and expires, the page URL does not.
        "source_url": media.source_url,
        "codec_name": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "r_frame_rate_value": r_rate,
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "avg_frame_rate_value": avg_rate,
        "fps": fps,
        "is_vfr": is_vfr,
        "nb_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "stream_duration": as_float(stream.get("duration")),
        "container_duration": as_float(container.get("duration")),
        # Metadata duration is the most trustworthy figure for a remote
        # video-only stream, whose header often omits it.
        "duration": as_float(container.get("duration")) or media.duration,
        "metadata_duration": media.duration,
    }
    _write_json_atomic(probe_file, probe)
    note = "VARIABLE frame rate" if is_vfr else (f"{fps:.3f} fps" if fps else "unknown fps")
    report(progress_callback, STAGE_PROBE,
           f"wrote {probe_file} -- {probe['width']}x{probe['height']}, {note}")
    return probe


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.core.audio")
    parser.add_argument("url")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audio-only", action="store_true", help="Skip the video probe")
    parser.add_argument("--probe-only", action="store_true", help="Skip the audio fetch")
    args = parser.parse_args(argv)

    try:
        media = resolve(args.url)
        wav = None if args.probe_only else ensure_audio(media, force=args.force)
        probe = None if args.audio_only else ensure_probe(media, force=args.force)
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    print(f"\nmedia_key : {media.media_key}")
    print(f"title     : {media.title}")
    print(f"duration  : {media.duration:.2f}s")
    if wav is not None:
        size = wav.stat().st_size
        expected = media.duration * config.AUDIO_BYTES_PER_SECOND
        # 16kHz mono 16-bit is exactly 32000 bytes/s, so this ratio should sit
        # very close to 1.000 -- the quickest check that the wav is intact.
        print(f"\naudio.wav : {wav}")
        print(f"  size    : {size:,} bytes ({size / 1_048_576:.2f} MiB)")
        print(f"  ratio   : {size / expected:.3f} of expected")
    if probe is not None:
        print(f"\nprobe.json: {paths.probe_path(media.media_key)}")
        print(f"  size    : {probe['width']}x{probe['height']} ({probe['codec_name']})")
        print(f"  fps     : {probe['fps']}   VFR: {probe['is_vfr']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
