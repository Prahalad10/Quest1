"""Progress reporting shim.

Every long-running function takes `progress_callback: ProgressCallback | None`.
Today the default implementation writes to stderr, which keeps stdout clean for
JSON output. A future SSE/web layer passes its own callback instead and needs no
changes in the core modules.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

# stage: short machine-ish label ("resolve", "asr"); message: human text.
ProgressCallback = Callable[[str, str], None]


def stderr_progress(stage: str, message: str) -> None:
    """Default sink: one line per event on stderr, flushed immediately."""
    print(f"[{stage}] {message}", file=sys.stderr, flush=True)


def report(callback: Optional[ProgressCallback], stage: str, message: str) -> None:
    """Call `callback` if given, else the stderr default.

    Pass an explicit no-op callback (``lambda *_: None``) to silence output.
    """
    (callback or stderr_progress)(stage, message)
