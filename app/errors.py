"""Exception hierarchy for the whole pipeline.

WHY A DEDICATED HIERARCHY
    Every failure this project raises deliberately is a Quest1Error subclass,
    carrying a message written to be shown to a user VERBATIM -- no stack trace,
    no internal jargon. That gives every caller one thing to catch:

        try:
            result = find_dialogue(url, text)
        except Quest1Error as exc:
            print(exc)          # CLI
            return 400, str(exc)  # API

    The CLI (app/cli.py) and the API (app/api.py) both rely on this: anything
    that is NOT a Quest1Error is a genuine bug and should surface as a real
    traceback rather than being dressed up as a friendly message.

THE RULE THESE ENFORCE
    Never swallow one of these to fall back to a guess. A wrong timestamp
    returned confidently is far worse than an honest failure -- the user cannot
    tell the difference between a right answer and a fabricated one.
"""

from __future__ import annotations


class Quest1Error(Exception):
    """Base class for every error this project raises deliberately.

    USED BY: app/cli.py and app/api.py as the single catch-all boundary between
    "expected failure, show the message" and "bug, let it crash".
    """


class InvalidInputError(Quest1Error):
    """The caller passed something malformed.

    Bad URL, empty dialogue text, a query too short to be meaningful, an unknown
    media_key, a negative timestamp.

    RAISED BY: resolve.py (_validate_url), matching.py (find_matches),
    paths.py (validate_media_key), frame.py, asr.py.
    MAPS TO: HTTP 400 in app/api.py.
    """


class ResolveError(Quest1Error):
    """yt-dlp could not extract usable metadata for the URL.

    A network failure, a removed video, an unsupported site, or a yt-dlp
    extractor that broke against a site change.

    RAISED BY: resolve.py. MAPS TO: HTTP 502 in app/api.py -- the failure is
    upstream, not in the request.
    """


class UnsupportedMediaError(Quest1Error):
    """The URL resolved, but the media is something we refuse to process.

    Playlists, live streams, upcoming premieres, DRM-protected content, videos
    over the duration cap, or a format set with no plain-HTTP stream to seek
    into. These are deliberate refusals, not failures.

    RAISED BY: resolve.py (_reject_unsupported, select_video_format).
    MAPS TO: HTTP 422 in app/api.py.
    """


class FFmpegError(Quest1Error):
    """An ffmpeg/ffprobe invocation failed, or the binary is missing.

    Always carries the tail of ffmpeg's stderr, because the first line of that
    output is usually the entire diagnosis.

    RAISED BY: core/ffmpeg.py. Caught and retried once by frame.py, which
    treats it as the signature of an expired signed URL.
    """


class AudioError(Quest1Error):
    """The audio track could not be fetched, transcoded, or validated.

    Most importantly: raised when the produced wav is SHORTER than the video by
    more than the tolerance, which means a truncated fetch. Silently accepting
    that would lose dialogue and produce confidently wrong "not found" answers.

    RAISED BY: core/audio.py.
    """
