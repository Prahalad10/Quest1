"""Tunable constants for the whole pipeline.

Nothing here is read at import time by anything that mutates state; every value
is overridable by an environment variable so behaviour can be changed without
editing code. Keep magic numbers OUT of the core modules and in here.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """Read an integer setting from the environment, failing loudly if malformed.

    WHY NOT int(os.environ.get(name, default)): a typo like QUEST1_MAX_DURATION=1h
    would raise a bare ValueError naming neither the variable nor its value. This
    reports both, so a misconfigured deployment is diagnosable from one line.

    USED BY: every integer setting defined in this module.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name}={raw!r} is not an integer") from exc


# --- Filesystem layout -------------------------------------------------------
# All per-video artifacts live under DATA_DIR/{media_key}/.
DATA_DIR = Path(os.environ.get("QUEST1_DATA_DIR", "data")).resolve()

# --- Input limits ------------------------------------------------------------
# Videos longer than this are rejected outright: ASR cost grows linearly and a
# multi-hour CPU transcription is never what the caller wanted.
MAX_VIDEO_DURATION_SECONDS = _env_int("QUEST1_MAX_DURATION", 3600)

# --- Format selection --------------------------------------------------------
# Frame extraction does ranged reads against the remote video stream, so cap the
# resolution: 4K buys nothing for a single PNG and costs a lot of bytes.
MAX_VIDEO_HEIGHT = _env_int("QUEST1_MAX_VIDEO_HEIGHT", 1080)

# --- Audio -------------------------------------------------------------------
# Whisper resamples everything to 16kHz mono internally, so producing it up front
# keeps the wav small and avoids a second resample at transcribe time.
AUDIO_SAMPLE_RATE = _env_int("QUEST1_AUDIO_SAMPLE_RATE", 16000)
AUDIO_CHANNELS = _env_int("QUEST1_AUDIO_CHANNELS", 1)

# 16-bit signed PCM => 2 bytes per sample per channel. Used to sanity-check the
# size of a produced wav against its expected duration.
AUDIO_BYTES_PER_SECOND = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * 2

# A produced wav shorter than this tolerance below the reported duration means a
# truncated fetch, which would silently lose dialogue. Fail instead.
AUDIO_DURATION_TOLERANCE_SECONDS = _env_int("QUEST1_AUDIO_TOLERANCE", 2)
AUDIO_DURATION_TOLERANCE_RATIO = 0.05

# --- Networking --------------------------------------------------------------
NETWORK_TIMEOUT_SECONDS = _env_int("QUEST1_NETWORK_TIMEOUT", 20)

# --- ASR ---------------------------------------------------------------------
# Model name passed to faster-whisper. "small" is the accuracy/speed sweet spot
# on CPU; set QUEST1_ASR_MODEL=base or tiny for faster (less accurate) runs, or
# medium/large-v3 if you have the time and RAM.
ASR_MODEL = os.environ.get("QUEST1_ASR_MODEL", "small")
ASR_DEVICE = os.environ.get("QUEST1_ASR_DEVICE", "cpu")
# int8 quantisation is ~2x faster than float32 on CPU with negligible WER cost.
ASR_COMPUTE_TYPE = os.environ.get("QUEST1_ASR_COMPUTE_TYPE", "int8")
ASR_BEAM_SIZE = _env_int("QUEST1_ASR_BEAM_SIZE", 5)
# None => auto-detect. Set QUEST1_ASR_LANGUAGE=en to skip detection and force it.
ASR_LANGUAGE = os.environ.get("QUEST1_ASR_LANGUAGE") or None
# Voice-activity detection trims silence before decoding: faster, and it stops
# Whisper hallucinating text over long quiet stretches.
ASR_VAD_FILTER = os.environ.get("QUEST1_ASR_VAD", "1") not in ("0", "false", "False")

# --- Index -------------------------------------------------------------------
# Bump this to invalidate every transcript index on disk. Any change to the
# normalization function, the word schema, or the ASR defaults must bump it,
# because a stale index would silently produce wrong offsets.
INDEX_VERSION = 1

# --- Matching thresholds -----------------------------------------------------
# All scores are rapidfuzz partial_ratio values in [0, 100].

# Below this, a span is not considered an occurrence at all.
MATCH_THRESHOLD = float(os.environ.get("QUEST1_MATCH_THRESHOLD", "70"))

# At or above this, a single occurrence is reported as "confident".
CONFIDENT_THRESHOLD = float(os.environ.get("QUEST1_CONFIDENT_THRESHOLD", "88"))

# If the runner-up occurrence scores within this margin of the best one, we
# cannot tell which the caller meant, so the result is downgraded to ambiguous.
AMBIGUITY_MARGIN = float(os.environ.get("QUEST1_AMBIGUITY_MARGIN", "5"))

# Safety valve for a very short query that matches everywhere.
MAX_OCCURRENCES = _env_int("QUEST1_MAX_OCCURRENCES", 20)

# When nothing clears MATCH_THRESHOLD, show the closest few spans instead --
# but only if they are at least this good, so we never present noise as a hint.
NEAR_MISS_THRESHOLD = float(os.environ.get("QUEST1_NEAR_MISS_THRESHOLD", "45"))
NEAR_MISS_COUNT = _env_int("QUEST1_NEAR_MISS_COUNT", 3)

# A query shorter than this many normalized characters is rejected: it would
# match almost anywhere and the resulting timestamp would be meaningless.
MIN_QUERY_CHARS = _env_int("QUEST1_MIN_QUERY_CHARS", 3)

# Confidence band names, used by the CLI and the future web layer.
BAND_CONFIDENT = "confident"
BAND_AMBIGUOUS = "ambiguous"
BAND_NO_MATCH = "no_match"

# --- Frame extraction --------------------------------------------------------
# Coarse -ss lands this far BEFORE the target so the decoder has a keyframe to
# start from; the remainder is consumed by a fine -ss after -i. Too small and
# the seek may miss the preceding keyframe; too large and it decodes needlessly.
FRAME_PREROLL_SECONDS = float(os.environ.get("QUEST1_FRAME_PREROLL", "5"))

# A signed CDN URL within this many seconds of expiry is re-resolved before use
# rather than being allowed to fail first.
FRAME_URL_EXPIRY_MARGIN_SECONDS = _env_int("QUEST1_FRAME_EXPIRY_MARGIN", 60)

# Single-frame extraction over HTTP should be quick; if it is not, something is
# wrong and failing beats hanging.
FRAME_TIMEOUT_SECONDS = _env_int("QUEST1_FRAME_TIMEOUT", 180)

# --- Resolve cache -----------------------------------------------------------
# Only used when a stream URL declares no expiry of its own; otherwise the URL's
# own expire= timestamp governs. Keeps a repeat query from re-running yt-dlp.
RESOLVE_CACHE_TTL_SECONDS = _env_int("QUEST1_RESOLVE_CACHE_TTL", 3600)

# --- Web / job layer ---------------------------------------------------------
# Finished jobs are kept this long so a browser that reconnects late can still
# read its result, then pruned so a long-running server does not grow forever.
JOB_RETENTION_SECONDS = _env_int("QUEST1_JOB_RETENTION", 3600)

# Hard cap on retained jobs, pruned oldest-finished-first. A second guard for
# the case where many jobs arrive inside the retention window.
JOB_MAX_RETAINED = _env_int("QUEST1_JOB_MAX_RETAINED", 200)

# How often the SSE endpoint checks for new progress events. Small enough to
# feel live, large enough not to spin a CPU core per connected browser.
SSE_POLL_INTERVAL_SECONDS = float(os.environ.get("QUEST1_SSE_POLL_INTERVAL", "0.2"))

# SSE comment sent when nothing has happened, so proxies do not close an idle
# connection during a long ASR stage.
SSE_KEEPALIVE_SECONDS = _env_int("QUEST1_SSE_KEEPALIVE", 15)

# --- ASR progress reporting --------------------------------------------------
# faster-whisper only yields a segment when it has finished decoding it, which on
# a long video can be 30-40s apart. Without a heartbeat between those events the
# progress bar sits frozen and looks broken. This is how often the heartbeat
# thread emits an interpolated estimate.
ASR_PROGRESS_INTERVAL_SECONDS = float(os.environ.get("QUEST1_ASR_PROGRESS_INTERVAL", "2"))

# --- Audio fetch strategy ----------------------------------------------------
# MEASURED: on a 797s video, ffmpeg reading the stream URL sequentially took
# 399s, while yt-dlp fetching the same bytes with chunked ranged requests took
# 8.7s -- a 30x difference. YouTube throttles one long sequential read to about
# 30 KB/s but serves chunked ranged requests at ~2 MB/s.
#
# This does NOT mean the video is downloaded. Only the audio track is fetched,
# and it was always written to disk as audio.wav regardless of transfer method.
# Frame extraction still reads the video remotely via HTTP ranges.
AUDIO_HTTP_CHUNK_SIZE = _env_int("QUEST1_AUDIO_CHUNK_SIZE", 10 * 1024 * 1024)
AUDIO_CONCURRENT_FRAGMENTS = _env_int("QUEST1_AUDIO_CONCURRENT_FRAGMENTS", 4)

# Set to "0" to force the old ffmpeg-streams-the-URL path. Kept because some
# hosts serve audio ffmpeg can read but yt-dlp cannot fetch as a file.
AUDIO_USE_CHUNKED_DOWNLOAD = os.environ.get("QUEST1_AUDIO_CHUNKED", "1") not in ("0", "false", "False")

