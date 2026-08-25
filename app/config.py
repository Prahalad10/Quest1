"""Tunable constants. Every value is env-overridable; keep magic numbers here.

Measurements behind the non-obvious defaults live in Constants.txt.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """int() from the environment, naming the variable when it is malformed."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name}={raw!r} is not an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name}={raw!r} is not a number") from exc


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in ("0", "false", "False")


DATA_DIR = Path(os.environ.get("QUEST1_DATA_DIR", "data")).resolve()

# --- Input limits ---
# A policy limit, not a promise of speed: at ~3x realtime a 2h video is ~40min
# of ASR on first index. Repeat queries on it are ~1s.
MAX_VIDEO_DURATION_SECONDS = _env_int("QUEST1_MAX_DURATION", 7200)
MAX_VIDEO_HEIGHT = _env_int("QUEST1_MAX_VIDEO_HEIGHT", 1080)
MIN_QUERY_CHARS = _env_int("QUEST1_MIN_QUERY_CHARS", 3)

# --- Audio ---
# 16kHz mono is what Whisper resamples to internally.
AUDIO_SAMPLE_RATE = _env_int("QUEST1_AUDIO_SAMPLE_RATE", 16000)
AUDIO_CHANNELS = _env_int("QUEST1_AUDIO_CHANNELS", 1)
AUDIO_BYTES_PER_SECOND = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * 2

# A wav short of the video duration means a truncated fetch, which would lose
# dialogue and return confident "not found" for the missing part.
AUDIO_DURATION_TOLERANCE_SECONDS = _env_int("QUEST1_AUDIO_TOLERANCE", 2)
AUDIO_DURATION_TOLERANCE_RATIO = 0.05

# Chunked ranged requests instead of one sequential ffmpeg read: 399s -> 8.7s
# on a 797s video, because YouTube throttles a single long read.
AUDIO_USE_CHUNKED_DOWNLOAD = _env_flag("QUEST1_AUDIO_CHUNKED", True)
AUDIO_HTTP_CHUNK_SIZE = _env_int("QUEST1_AUDIO_CHUNK_SIZE", 10 * 1024 * 1024)
AUDIO_CONCURRENT_FRAGMENTS = _env_int("QUEST1_AUDIO_CONCURRENT_FRAGMENTS", 4)

# None = the ORIGINAL track. Multi-language uploads carry dubs, often at a
# HIGHER bitrate than the original; picking on bitrate alone transcribed an
# English video into Arabic and made every query miss.
AUDIO_TRACK_LANGUAGE = os.environ.get("QUEST1_AUDIO_LANGUAGE") or None

# --- Networking ---
NETWORK_TIMEOUT_SECONDS = _env_int("QUEST1_NETWORK_TIMEOUT", 20)

# ok.ru resets a share of connections; the same URL fails then succeeds. Only
# transport-shaped errors retry -- see resolve.is_transient_error.
RESOLVE_MAX_ATTEMPTS = _env_int("QUEST1_RESOLVE_ATTEMPTS", 8)
RESOLVE_RETRY_BACKOFF_SECONDS = _env_float("QUEST1_RESOLVE_BACKOFF", 2.0)
RESOLVE_HTTP_RETRIES = _env_int("QUEST1_RESOLVE_HTTP_RETRIES", 3)
RESOLVE_CACHE_TTL_SECONDS = _env_int("QUEST1_RESOLVE_CACHE_TTL", 3600)

# --- ASR ---
ASR_MODEL = os.environ.get("QUEST1_ASR_MODEL", "small")
ASR_DEVICE = os.environ.get("QUEST1_ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.environ.get("QUEST1_ASR_COMPUTE_TYPE", "int8")
ASR_LANGUAGE = os.environ.get("QUEST1_ASR_LANGUAGE") or None
ASR_VAD_FILTER = _env_flag("QUEST1_ASR_VAD", True)

# Greedy decoding: 2.95x faster than beam 5 (76.4s -> 25.9s on the same audio).
# Matching is fuzzy at MATCH_THRESHOLD, so the accuracy beam search buys does
# not change whether a line is found.
ASR_BEAM_SIZE = _env_int("QUEST1_ASR_BEAM_SIZE", 1)

# faster-whisper only yields a segment once decoded, 30-40s apart on long
# audio, so a bar driven by segment arrivals alone freezes. See asr._Heartbeat.
ASR_PROGRESS_INTERVAL_SECONDS = _env_float("QUEST1_ASR_PROGRESS_INTERVAL", 2.0)

# --- Index ---
# Bump to invalidate every index on disk. Required after ANY change to
# normalize.py or the word schema: a stale index yields wrong offsets silently.
INDEX_VERSION = 2

# --- Matching (rapidfuzz partial_ratio, 0-100) ---
# 70 is validated: 0/184 false matches at 70, 3/184 at 65. Do not lower.
MATCH_THRESHOLD = _env_float("QUEST1_MATCH_THRESHOLD", 70.0)
CONFIDENT_THRESHOLD = _env_float("QUEST1_CONFIDENT_THRESHOLD", 88.0)
# Runner-up within this margin -> ambiguous, rather than a coin-flip reported
# as fact.
AMBIGUITY_MARGIN = _env_float("QUEST1_AMBIGUITY_MARGIN", 5.0)
MAX_OCCURRENCES = _env_int("QUEST1_MAX_OCCURRENCES", 20)
NEAR_MISS_THRESHOLD = _env_float("QUEST1_NEAR_MISS_THRESHOLD", 45.0)
NEAR_MISS_COUNT = _env_int("QUEST1_NEAR_MISS_COUNT", 3)

BAND_CONFIDENT = "confident"
BAND_AMBIGUOUS = "ambiguous"
BAND_NO_MATCH = "no_match"

# --- Frame extraction ---
# Coarse -ss lands this far before the target so the decoder has a keyframe;
# the remainder is a fine -ss after -i.
FRAME_PREROLL_SECONDS = _env_float("QUEST1_FRAME_PREROLL", 5.0)
FRAME_URL_EXPIRY_MARGIN_SECONDS = _env_int("QUEST1_FRAME_EXPIRY_MARGIN", 60)
FRAME_TIMEOUT_SECONDS = _env_int("QUEST1_FRAME_TIMEOUT", 180)

# --- Web / jobs ---
JOB_RETENTION_SECONDS = _env_int("QUEST1_JOB_RETENTION", 3600)
JOB_MAX_RETAINED = _env_int("QUEST1_JOB_MAX_RETAINED", 200)
SSE_POLL_INTERVAL_SECONDS = _env_float("QUEST1_SSE_POLL_INTERVAL", 0.2)
SSE_KEEPALIVE_SECONDS = _env_int("QUEST1_SSE_KEEPALIVE", 15)
