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
# Videos longer than this are rejected outright. Raised from 3600 to 7200 so a
# feature-length film (typically 90-120 minutes) is accepted; the parallel ASR
# path below is what makes that duration tractable.
MAX_VIDEO_DURATION_SECONDS = _env_int("QUEST1_MAX_DURATION", 7200)

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

# --- Audio track selection ---------------------------------------------------
# Which audio track to transcribe on a multi-language upload. None = the
# ORIGINAL track, which is what you almost always want.
#
# WHY THIS EXISTS: YouTube multi-language uploads carry the original plus a set
# of dubs, and dubs are often encoded at a HIGHER bitrate than the original.
# Choosing on bitrate alone picked an Arabic dub of an English video, which
# transcribed cleanly into Arabic and made every English query return
# "not found" with no error to explain it.
#
# Set to a language code (e.g. "hi") to transcribe that dub on purpose. It is
# honoured only when a track in that language exists.
AUDIO_TRACK_LANGUAGE = os.environ.get("QUEST1_AUDIO_LANGUAGE") or None

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
# Greedy decoding. THE SINGLE BIGGEST SPEED LEVER IN THE PIPELINE.
#
# MEASURED on 90s of dense speech (small, int8, 8 threads), changing only this:
#
#     beam_size=5   76.4s   1.18x realtime
#     beam_size=1   25.9s   3.47x realtime      <- 2.95x faster
#
# Beam search runs the decoder once per beam and keeps the best path, so beam 5
# is close to five times the decoder work for a marginal accuracy gain. That
# accuracy gain does not survive contact with this problem: matching is fuzzy
# (rapidfuzz partial_ratio at MATCH_THRESHOLD=70), so a transcript has to be
# wrong by far more than a beam-1/beam-5 disagreement before a line stops being
# found. Validated by scripts/test_matrix.py against known timestamps.
#
# Set QUEST1_ASR_BEAM_SIZE=5 to trade the speed back for transcript quality.
ASR_BEAM_SIZE = _env_int("QUEST1_ASR_BEAM_SIZE", 1)
# None => auto-detect. Set QUEST1_ASR_LANGUAGE=en to skip detection and force it.
ASR_LANGUAGE = os.environ.get("QUEST1_ASR_LANGUAGE") or None
# Voice-activity detection trims silence before decoding: faster, and it stops
# Whisper hallucinating text over long quiet stretches.
ASR_VAD_FILTER = os.environ.get("QUEST1_ASR_VAD", "1") not in ("0", "false", "False")

# --- Parallel ASR ------------------------------------------------------------
# OFF BY DEFAULT, because it was measured and it does not pay on this machine.
#
# The idea: one Whisper stream cannot use four cores, so decode different pieces
# of audio at once. Thread scaling at beam=5 predicted a 2.5x win (see
# ASR_WORKER_THREADS below). END-TO-END IT DELIVERED ALMOST NOTHING:
#
#     529s audio   serial 134.6s (3.93x)   3 workers 121.1s (4.37x)   1.11x
#     716s audio   serial 252.9s (2.83x)   4 workers 244.6s (2.93x)   1.03x
#
# WHY THE PREDICTION FAILED: the thread-scaling table was measured at beam=5,
# where decoding is compute-bound and idle cores really are idle. Setting
# beam=1 removes that bottleneck and the workload becomes memory-BANDWIDTH
# bound -- four processes then contend for one memory bus instead of four
# separate compute units. Each worker ran at ~0.73x realtime instead of the
# ~3.4x it manages alone.
#
# It is kept, working and seam-tested, because the trade changes on hardware
# with more memory bandwidth or real physical cores. It costs one model copy
# per worker (~0.5 GB), which is why it is not on by default on a 7.8 GB
# machine for a 5% gain. Set QUEST1_ASR_PARALLEL=1 to enable it.
ASR_PARALLEL = os.environ.get("QUEST1_ASR_PARALLEL", "0") not in ("0", "false", "False")

# Audio shorter than this stays on the single-process path: each worker pays a
# model load of a few seconds, which would dominate a short clip.
ASR_PARALLEL_MIN_SECONDS = _env_int("QUEST1_ASR_PARALLEL_MIN", 90)

# Length of the region each chunk OWNS. Smaller chunks balance the pool better
# on a video whose speech is unevenly distributed, but every chunk restarts the
# decoder cold, so very small chunks cost accuracy at the seams.
ASR_CHUNK_SECONDS = float(os.environ.get("QUEST1_ASR_CHUNK_SECONDS", "180"))

# Extra audio decoded either side of a chunk purely as context, then discarded.
# Long enough to contain a whole spoken sentence so the decoder is never started
# mid-phrase; output from it is never kept, so this cannot duplicate a word.
ASR_CHUNK_OVERLAP_SECONDS = float(os.environ.get("QUEST1_ASR_CHUNK_OVERLAP", "6"))

# Upper bound on worker processes. Each worker holds its own copy of the model,
# so this is a MEMORY limit as much as a CPU one -- this machine has 7.8 GB, and
# a "small" int8 model costs roughly 0.5 GB resident per worker.
ASR_MAX_WORKERS = _env_int("QUEST1_ASR_MAX_WORKERS", 4)

# Threads given to EACH worker. One, deliberately.
#
# MEASURED on the same 90s of dense speech (small, int8, beam 5), varying only
# the thread count given to a single stream:
#
#     8 threads  76.4s  1.18x realtime      <- slower than 4: this machine has
#     4 threads  69.4s  1.30x realtime         4 physical cores, and the extra
#     2 threads  82.2s  1.10x realtime         hyperthreads only add contention
#     1 thread  111.1s  0.81x realtime
#
# Throughput barely moves between 1 and 4 threads, so a single stream cannot
# use the machine: 4 threads buys only 1.6x over 1 thread. Running FOUR
# one-thread workers instead gives 4 x 0.81 = 3.24x realtime aggregate, versus
# 1.30x for one four-thread stream -- a 2.5x speedup from the same cores.
#
# More than one thread per worker would also oversubscribe: 4 workers x 2
# threads is 8 threads on 4 physical cores, which makes every worker slower
# without finishing any of them sooner.
ASR_WORKER_THREADS = _env_int("QUEST1_ASR_WORKER_THREADS", 1)

# --- Index -------------------------------------------------------------------
# Bump this to invalidate every transcript index on disk. Any change to the
# normalization function, the word schema, or the ASR defaults must bump it,
# because a stale index would silently produce wrong offsets.
INDEX_VERSION = 2   # v2: ASR_BEAM_SIZE default 5 -> 1, parallel chunked decoding

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

