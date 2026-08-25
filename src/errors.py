"""Error hierarchy. Every deliberate failure carries a user-facing message.

Callers catch DialogueFrameError and print it verbatim. Anything else is a real bug
and should surface as a traceback rather than be dressed up as a friendly
message. Never swallow one of these to fall back to a guess: a confidently
wrong timestamp is worse than an honest failure.
"""

from __future__ import annotations


class DialogueFrameError(Exception):
    """Base for every expected failure. The catch-all in cli.py and api.py."""


class InvalidInputError(DialogueFrameError):
    """Malformed input: bad URL, empty query, unknown media_key. HTTP 400."""


class ResolveError(DialogueFrameError):
    """yt-dlp could not extract usable metadata. Upstream failure. HTTP 502."""


class UnsupportedMediaError(DialogueFrameError):
    """Resolved, but refused: playlist, live, DRM, over the duration cap. HTTP 422."""


class FFmpegError(DialogueFrameError):
    """ffmpeg/ffprobe failed or is missing. Carries the tail of its stderr.

    frame.py treats this as the signature of an expired signed URL and retries
    once after re-resolving.
    """


class AudioError(DialogueFrameError):
    """Audio could not be fetched, transcoded, or validated.

    Notably raised when the wav is shorter than the video beyond tolerance,
    which means a truncated fetch -- accepting that would lose dialogue and
    produce confident "not found" answers for the missing part.
    """
