"""Progress reporting shim shared by every long-running stage.

WHY THIS EXISTS
    The pipeline has stages that take anywhere from milliseconds (match) to
    minutes (ASR on a long video). A CLI user wants those on stderr; a web user
    wants them streamed to a browser over SSE. Rather than teach every core
    module about both, each one simply calls `report(...)` and the *caller*
    decides where the events go by passing a `progress_callback`.

    That means app/core/* never imports anything web-related, and the same code
    serves `python -m app.cli` and the FastAPI layer in app/api.py.

THE CALLBACK CONTRACT
    progress_callback(stage: str, percent: float | None, message: str)

    stage    short machine-readable label -- "resolve", "audio", "asr",
             "match", "frame". The web UI keys its stage list off these.
    percent  0-100 progress WITHIN that stage, or None when a stage cannot
             know its own progress (a yt-dlp metadata fetch, for example).
             app/service.py converts these into an overall percentage.
    message  human-readable detail, safe to display verbatim.

USED BY
    Every module in app/core/, plus app/service.py which wraps a caller's
    callback to add overall-progress arithmetic.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

# The signature every progress consumer must accept. Declared once here so the
# core modules, the service layer and the API all agree on the shape.
ProgressCallback = Callable[[str, Optional[float], str], None]


def stderr_progress(stage: str, percent: Optional[float], message: str) -> None:
    """Default sink: one line per event on stderr.

    WHY STDERR: stdout is reserved for the actual result, so `--json` output
    stays machine-parseable even while progress is streaming. Flushed on every
    call so progress appears live rather than in a buffered burst at the end.

    USED BY: `report()` whenever the caller passed no callback of its own --
    which is the normal case for the CLI.
    """
    prefix = f"[{stage}]"
    if percent is None:
        print(f"{prefix} {message}", file=sys.stderr, flush=True)
    else:
        print(f"{prefix} {percent:5.1f}%  {message}", file=sys.stderr, flush=True)


def report(
    callback: Optional[ProgressCallback],
    stage: str,
    message: str,
    percent: Optional[float] = None,
) -> None:
    """Emit one progress event to `callback`, or to stderr if none was given.

    WHY THE ARGUMENT ORDER DIFFERS FROM THE CALLBACK: `percent` is optional and
    most call sites do not have one, so it sits last here for convenience. The
    callback itself is always invoked in the documented (stage, percent,
    message) order.

    WHY IT SWALLOWS CALLBACK ERRORS: a progress consumer is a side channel. If
    a web client disconnects mid-stream and its callback raises, the actual
    pipeline work must still finish rather than dying for want of a listener.

    USED BY: every stage in app/core/, and app/service.py.
    """
    sink = callback or stderr_progress
    try:
        sink(stage, percent, message)
    except Exception:  # noqa: BLE001 - progress must never break the pipeline
        pass


def null_progress(stage: str, percent: Optional[float], message: str) -> None:
    """A callback that discards everything.

    USED BY: `python -m app.cli --quiet`, and any test that does not want
    progress noise in its captured output.
    """
    return None
