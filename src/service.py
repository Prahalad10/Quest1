"""The pipeline: find_dialogue() is the one entry point. Prints nothing.

    resolve -> audio -> probe -> index -> match -> frame

Indexing (resolve/audio/probe/index) is expensive and cached per video.
Querying (match/frame) is cheap. That split is why a second search on the same
video skips ASR entirely.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src import config, paths
from src.core import frame as frame_mod
from src.core.audio import ensure_audio, ensure_probe
from src.core.index import ensure_index
from src.core.matching import MatchResult, find_matches, format_timestamp
from src.core.resolve import ResolvedMedia, resolve_cached
from src.progress import ProgressCallback, report, stderr_progress

STAGE = "pipeline"

# Relative cost of each stage on a COLD run, used to turn per-stage progress
# into one overall percentage.
#
# CALIBRATED FROM MEASUREMENT: ASR is ~95% of cold wall time. The original
# weights gave it 50 of 100, so the bar spent half its range on 95% of the
# work -- it crawled, then leapt to 100.
#
# index_check and index are the same module either side of ASR, and are listed
# either side of it for that reason. Collapsing them put the cache-miss message
# at the index stage's offset which, because the overall percentage never moves
# backwards, pinned the bar near 100% for the entire transcription.
STAGE_WEIGHTS: dict[str, float] = {
    "resolve": 3.0,
    "audio": 5.0,
    "probe": 2.0,
    "index_check": 0.5,   # cache lookup, BEFORE any transcription
    "asr": 85.0,
    "index": 1.5,         # index assembly, AFTER ASR finishes
    "match": 1.0,
    "frame": 2.0,
}

STAGE_ORDER: list[str] = ["resolve", "audio", "probe", "index_check", "asr",
                          "index", "match", "frame"]


def _stage_base(stage: str) -> float:
    """Cumulative weight of every stage before `stage`."""
    total = 0.0
    for name in STAGE_ORDER:
        if name == stage:
            return total
        total += STAGE_WEIGHTS.get(name, 0.0)
    return total


class _OverallProgress:
    """Wraps a caller's callback to add a single monotonic overall percentage.

    Core modules only know their own progress -- asr.py can say "60% of the
    audio transcribed" but has no idea it is one stage of six.
    """

    def __init__(self, inner: Optional[ProgressCallback]) -> None:
        # None means "use the default sink", not "discard" -- pass
        # progress.null_progress explicitly to silence output.
        self._inner: ProgressCallback = inner if inner is not None else stderr_progress
        self._highest = 0.0
        # The ASR heartbeat reports from its own thread while the pipeline
        # thread reports stage transitions, so this is genuinely concurrent.
        self._lock = threading.Lock()
        self.seen: list[str] = []

    def __call__(self, stage: str, percent: Optional[float], message: str) -> None:
        with self._lock:
            if stage not in self.seen:
                self.seen.append(stage)

        # percent=None means "this stage cannot measure itself", NOT "finished".
        # Treating it as 100 would pin the bar to the stage's END on its first
        # message, and the monotonic clamp would then block every real reading.
        within = 0.0 if percent is None else max(0.0, min(100.0, percent))
        overall = _stage_base(stage) + STAGE_WEIGHTS.get(stage, 0.0) * within / 100.0

        with self._lock:
            self._highest = max(self._highest, min(overall, 99.0))
            current = self._highest
        self._inner(stage, round(current, 1), message)

    def complete(self, message: str = "done") -> None:
        """Final 100%, so a UI can close its bar rather than leave it at 99%."""
        with self._lock:
            self._highest = 100.0
        self._inner("done", 100.0, message)


@dataclass
class DialogueResult:
    """The complete answer to one query, as data rather than printed text."""

    status: str                        # found | not_found
    media_key: str
    title: str
    source_url: str
    query: str
    # --- populated only when status == "found" ---
    timestamp: Optional[str] = None
    timestamp_seconds: Optional[float] = None
    frame_number: Optional[int] = None   # None for VFR -- see frame_note
    frame_pts: Optional[float] = None
    matched_text: Optional[str] = None
    image_path: Optional[str] = None
    score: Optional[float] = None
    band: str = config.BAND_NO_MATCH
    context: Optional[str] = None
    frame_note: Optional[str] = None
    other_occurrences: list[dict[str, Any]] = field(default_factory=list)
    near_misses: list[dict[str, Any]] = field(default_factory=list)
    cached_stages: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "media_key": self.media_key,
            "title": self.title,
            "source_url": self.source_url,
            "query": self.query,
            "timestamp": self.timestamp,
            "timestamp_seconds": self.timestamp_seconds,
            "frame_number": self.frame_number,
            "frame_pts": self.frame_pts,
            "matched_text": self.matched_text,
            "image_path": self.image_path,
            "score": self.score,
            "band": self.band,
            "context": self.context,
            "frame_note": self.frame_note,
            "other_occurrences": self.other_occurrences,
            "near_misses": self.near_misses,
            "cached_stages": self.cached_stages,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def find_dialogue(
    url: str,
    text: str,
    *,
    force: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> DialogueResult:
    """Locate the frame where `text` is spoken in the video at `url`.

    Raises DialogueFrameError subclasses whose messages are written to be shown to a
    user verbatim. A not_found result carries near_misses so the caller can show
    what the transcript actually says instead of fabricating an answer.
    """
    started = time.time()
    progress = _OverallProgress(progress_callback)
    cached: list[str] = []

    media: ResolvedMedia
    media, resolve_hit = resolve_cached(url, force=force, progress_callback=progress)
    if resolve_hit:
        cached.append("resolve")

    # Existence is checked BEFORE ensure_*, because those return the same value
    # whether they built or reused; looking first is the only honest way to
    # report which stages were cached.
    audio_existed = not force and paths.audio_path(media.media_key).exists()
    wav = ensure_audio(media, force=force, progress_callback=progress)
    if audio_existed:
        cached.append("audio")

    probe_existed = not force and paths.probe_path(media.media_key).exists()
    probe = ensure_probe(media, force=force, progress_callback=progress)
    if probe_existed:
        cached.append("probe")

    index = ensure_index(media.media_key, wav, force=force, progress_callback=progress)
    if index.from_cache:
        cached.append("index")

    match: MatchResult = find_matches(index, text, progress_callback=progress)

    if not match.found:
        report(progress, STAGE, "no match above threshold; returning near misses")
        progress.complete("no match")
        return DialogueResult(
            status="not_found", media_key=media.media_key, title=media.title,
            source_url=media.source_url, query=text, band=match.band,
            near_misses=[o.to_dict() for o in match.near_misses],
            cached_stages=cached, elapsed_seconds=time.time() - started,
        )

    best = match.best
    assert best is not None  # guaranteed by match.found

    # best.start_time is when the matched word BEGINS being spoken, which is the
    # honest answer to "where is this line spoken". Nudging into the word would
    # look better on fast speech but would no longer be the frame it starts on.
    frame_path = paths.frames_dir(media.media_key) / frame_mod.frame_filename(best.start_time)
    frame_existed = not force and frame_path.exists()
    frame_result = frame_mod.extract_frame(media, best.start_time, probe,
                                           force=force, progress_callback=progress)
    if frame_existed:
        cached.append("frame")

    progress.complete(f"found at {format_timestamp(best.start_time)}")
    return DialogueResult(
        status="found", media_key=media.media_key, title=media.title,
        source_url=media.source_url, query=text,
        timestamp=format_timestamp(best.start_time),
        timestamp_seconds=best.start_time,
        frame_number=frame_result.frame_number,
        frame_pts=frame_result.frame_pts,
        matched_text=best.matched_text,
        image_path=frame_result.path,
        score=best.score, band=match.band, context=best.context,
        frame_note=frame_result.note,
        # Every other place the line was said, so an ambiguous result can be
        # judged by a human rather than silently resolved by us.
        other_occurrences=[o.to_dict() for o in match.occurrences if o is not best],
        cached_stages=cached, elapsed_seconds=time.time() - started,
    )
