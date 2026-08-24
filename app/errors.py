"""Exception hierarchy.

Every failure the pipeline can produce is one of these, carrying a message that
is safe and useful to show a user verbatim. Never swallow one of these to fall
back to a guess -- a wrong answer is worse than a loud failure here.
"""

from __future__ import annotations


class Quest1Error(Exception):
    """Base class for every error this project raises deliberately."""


class InvalidInputError(Quest1Error):
    """The caller passed something malformed (bad URL, empty text, ...)."""


class ResolveError(Quest1Error):
    """yt-dlp could not extract usable metadata for the URL."""


class UnsupportedMediaError(Quest1Error):
    """The URL resolved, but the media is something we refuse to process.

    Playlists, live streams, DRM-protected content, over-long videos, or a
    format set with no plain-HTTP stream to seek into.
    """


class FFmpegError(Quest1Error):
    """An ffmpeg/ffprobe invocation failed, or the binary is missing."""


class AudioError(Quest1Error):
    """The audio track could not be fetched, transcoded, or validated."""
