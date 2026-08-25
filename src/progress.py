"""Progress callback shared by every stage.

Core modules call report(); the CALLER decides where events go by passing a
callback. That is why src/core/* never imports anything web-related and the
same code serves the CLI and the SSE endpoint.

Contract: callback(stage, percent, message). percent is 0-100 within that
stage, or None when a stage cannot know its own progress; service.py turns
those into one overall percentage.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

ProgressCallback = Callable[[str, Optional[float], str], None]


def stderr_progress(stage: str, percent: Optional[float], message: str) -> None:
    """Default sink. stderr, so stdout stays parseable for --json."""
    head = f"[{stage}]"
    body = message if percent is None else f"{percent:5.1f}%  {message}"
    print(f"{head} {body}", file=sys.stderr, flush=True)


def null_progress(stage: str, percent: Optional[float], message: str) -> None:
    """Discards everything. Used by --quiet."""


def report(
    callback: Optional[ProgressCallback],
    stage: str,
    message: str,
    percent: Optional[float] = None,
) -> None:
    """Emit one event, defaulting to stderr.

    Callback errors are swallowed: a disconnected web client must not kill the
    pipeline work it was watching.
    """
    try:
        (callback or stderr_progress)(stage, percent, message)
    except Exception:  # noqa: BLE001 - progress must never break the pipeline
        pass
