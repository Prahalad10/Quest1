"""Thin, loud wrapper around the ffmpeg and ffprobe binaries.

Every subprocess call in the project goes through here so that:
  * a missing binary produces one clear message, not a bare FileNotFoundError,
  * a non-zero exit always surfaces the tail of ffmpeg's stderr,
  * remote HTTP inputs get identical header/reconnect handling everywhere.

Used by audio.py (Step 2) and frame.py (Step 5).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Optional

from app import config
from app.errors import FFmpegError
from app.progress import ProgressCallback, report

# Applied to every remote HTTP input: a dropped connection mid-fetch should
# retry rather than silently truncate the output.
HTTP_RECONNECT_ARGS = [
    "-reconnect", "1",
    "-reconnect_streamed", "1",
    "-reconnect_delay_max", "5",
]


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(
            f"{name} was not found on PATH. Install ffmpeg (it ships both ffmpeg and "
            f"ffprobe) and reopen your terminal. Run `python scripts/check_env.py` to confirm."
        )
    return path


def ffmpeg_binary() -> str:
    return _binary("ffmpeg")


def ffprobe_binary() -> str:
    return _binary("ffprobe")


def build_input_headers(http_headers: Optional[dict[str, str]]) -> list[str]:
    """Turn yt-dlp's per-format headers into ffmpeg input arguments.

    User-Agent gets its own flag because some ffmpeg builds ignore it when it is
    folded into the generic -headers blob.
    """
    headers = {k: v for k, v in (http_headers or {}).items() if v}
    args: list[str] = []

    user_agent = None
    for key in list(headers):
        if key.lower() == "user-agent":
            user_agent = headers.pop(key)
    if user_agent:
        args += ["-user_agent", user_agent]

    if headers:
        blob = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        args += ["-headers", blob]
    return args


def _tail(text: str, lines: int = 12) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return "(no stderr output)"
    return "\n".join(stripped.splitlines()[-lines:])


def run_ffprobe_json(args: list[str], timeout: int = 120) -> dict[str, Any]:
    """Run ffprobe with JSON output and return the parsed document."""
    cmd = [ffprobe_binary(), "-v", "error", "-print_format", "json", *args]
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

    `-progress pipe:1` gives machine-readable progress on stdout, so this works
    for any invocation that writes its real output to a file.
    """
    cmd = [
        ffmpeg_binary(),
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-nostats",
        "-progress", "pipe:1",
        *args,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise FFmpegError(f"Could not run ffmpeg: {exc}") from exc

    last_percent = -10.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key != "out_time_us" or not value.isdigit():
                continue
            seconds = int(value) / 1_000_000
            if total_duration and total_duration > 0:
                percent = min(100.0, seconds / total_duration * 100)
                if percent - last_percent >= 10:
                    last_percent = percent
                    report(
                        progress_callback, stage,
                        f"{percent:5.1f}%  ({seconds:.1f}s / {total_duration:.1f}s)",
                    )
            elif seconds - last_percent >= 30:
                last_percent = seconds
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
    """ffprobe a local path or remote URL, returning streams + format sections."""
    args: list[str] = []
    if url.startswith(("http://", "https://")):
        args += build_input_headers(http_headers)
    if select_streams:
        args += ["-select_streams", select_streams]
    args += ["-show_streams", "-show_format", url]
    return run_ffprobe_json(args, timeout=timeout)
