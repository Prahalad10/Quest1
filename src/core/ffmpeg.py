"""Wrapper around the ffmpeg/ffprobe binaries.

Every subprocess call goes through here so a missing binary gives one clear
message, a non-zero exit always surfaces ffmpeg's stderr tail, and remote HTTP
inputs get identical header and reconnect handling.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Optional

from src import config
from src.errors import FFmpegError
from src.progress import ProgressCallback, report

# A dropped connection mid-fetch should retry, not silently truncate.
HTTP_RECONNECT_ARGS = ["-reconnect", "1", "-reconnect_streamed", "1",
                       "-reconnect_delay_max", "5"]


def binary(name: str) -> str:
    """ffmpeg is a system dependency pip cannot install, so say so explicitly."""
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(
            f"{name} was not found on PATH. Install ffmpeg (it ships both ffmpeg and "
            f"ffprobe) and reopen your terminal. Run `python scripts/check_env.py` to confirm."
        )
    return path


def build_input_headers(http_headers: Optional[dict[str, str]]) -> list[str]:
    """yt-dlp per-format headers -> ffmpeg input args.

    The CDN rejects requests whose User-Agent does not match the session the
    signed URL was issued to, so a missing header reads as 403 rather than as
    an auth problem. User-Agent needs its own flag because some ffmpeg builds
    ignore it inside the generic -headers blob.
    """
    headers = {k: v for k, v in (http_headers or {}).items() if v}
    args: list[str] = []
    user_agent = next((headers.pop(k) for k in list(headers) if k.lower() == "user-agent"), None)
    if user_agent:
        args += ["-user_agent", user_agent]
    if headers:
        args += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    return args


def _tail(text: str, lines: int = 12) -> str:
    """ffmpeg can emit hundreds of lines; the cause is in the last few."""
    stripped = (text or "").strip()
    return "\n".join(stripped.splitlines()[-lines:]) if stripped else "(no stderr output)"


def run_ffprobe_json(args: list[str], timeout: int = 120) -> dict[str, Any]:
    """Run ffprobe and parse its JSON. Any failure means the stream is unreadable."""
    cmd = [binary("ffprobe"), "-v", "error", "-print_format", "json", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffprobe timed out after {timeout}s.") from exc
    except OSError as exc:
        raise FFmpegError(f"Could not run ffprobe: {exc}") from exc

    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe exited {proc.returncode}:\n{_tail(proc.stderr)}")
    if not proc.stdout.strip():
        raise FFmpegError("ffprobe produced no output (the stream may be unreadable).")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned unparseable JSON: {exc}") from exc


def run_ffmpeg(
    args: list[str],
    *,
    total_duration: Optional[float] = None,
    stage: str = "ffmpeg",
    progress_callback: Optional[ProgressCallback] = None,
    timeout: int = 3600,
) -> None:
    """Run ffmpeg, streaming progress and raising FFmpegError on failure.

    `-progress pipe:1` gives machine-readable progress on stdout, which works
    because every invocation here writes its real output to a file. Throttled
    to ~10% steps so a long transcode does not flood the progress stream.
    """
    cmd = [binary("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error",
           "-nostats", "-progress", "pipe:1", *args]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
    except OSError as exc:
        raise FFmpegError(f"Could not run ffmpeg: {exc}") from exc

    last = -10.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key != "out_time_us" or not value.isdigit():
                continue
            seconds = int(value) / 1_000_000
            if total_duration and total_duration > 0:
                percent = min(100.0, seconds / total_duration * 100)
                if percent - last >= 10:
                    last = percent
                    report(progress_callback, stage,
                           f"{seconds:.1f}s / {total_duration:.1f}s", percent=percent)
            elif seconds - last >= 30:
                last = seconds  # no known duration: a running total, not a ratio
                report(progress_callback, stage, f"processed {seconds:.1f}s")
        stderr = proc.communicate(timeout=timeout)[1]
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise FFmpegError(f"ffmpeg timed out after {timeout}s.") from exc

    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg exited {proc.returncode}:\n{_tail(stderr)}")


def probe_media(
    url: str,
    *,
    http_headers: Optional[dict[str, str]] = None,
    select_streams: Optional[str] = None,
    timeout: int = config.NETWORK_TIMEOUT_SECONDS * 6,
) -> dict[str, Any]:
    """ffprobe a local path or remote URL.

    For a URL this reads only the container header via range requests, which is
    why probing a 1080p feature takes about a second.
    """
    args: list[str] = []
    if url.startswith(("http://", "https://")):
        args += build_input_headers(http_headers)
    if select_streams:
        args += ["-select_streams", select_streams]
    args += ["-show_streams", "-show_format", url]
    return run_ffprobe_json(args, timeout=timeout)
