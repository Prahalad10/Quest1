"""The pipeline as one callable function, independent of how it is invoked.

WHY THIS MODULE EXISTS
    Step 6 put the orchestration inside app/cli.py, which made it reachable only
    from a terminal. Everything above the core modules -- the CLI, the FastAPI
    layer in app/api.py, and anything added later -- needs the SAME sequence of
    stages with the SAME caching behaviour. Duplicating that sequence would
    guarantee the two copies drift apart.

    So the orchestration lives here, returns structured data, and prints
    NOTHING. app/cli.py formats it for a terminal; app/api.py serialises it to
    JSON. Neither re-implements the pipeline.

THE STAGES
    resolve  yt-dlp metadata + signed stream URLs      cached per URL
    audio    audio-only fetch -> 16kHz mono wav        cached per media_key
    probe    ffprobe of the video stream (fps, VFR)    cached per media_key
    index    faster-whisper ASR + word timestamps      cached per media_key
    match    fuzzy search over the transcript          never cached (it is ms)
    frame    ranged HTTP seek -> PNG                   cached per timestamp

    Indexing is expensive and per-video; querying is cheap and per-text. That
    split is the reason a second search on the same video skips ASR entirely.

USED BY
    app/cli.py (thin terminal wrapper) and app/api.py (Step 8).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app import config, paths
from app.core import frame as frame_mod
from app.core.audio import ensure_audio, ensure_probe
from app.core.index import ensure_index
from app.core.matching import MatchResult, find_matches, format_timestamp
from app.core.resolve import ResolvedMedia, resolve_cached
from app.progress import ProgressCallback, report, stderr_progress

STAGE = "pipeline"

# Relative cost of each stage on a COLD run, used to turn per-stage progress
# into a single overall percentage for a progress bar. ASR dominates by design:
# on CPU it runs at roughly 0.7x realtime while everything else is seconds.
# These are weights, not measurements -- they only need to be roughly right for
# the bar to move smoothly rather than lurch.
STAGE_WEIGHTS: dict[str, float] = {
    "resolve": 5.0,
    "audio": 20.0,
    "probe": 5.0,
    "asr": 50.0,      # emitted by app/core/asr.py during transcription
    "index": 5.0,     # index assembly after ASR finishes
    "match": 5.0,
    "frame": 10.0,
}

# Order matters: the base offset of a stage is the sum of every earlier weight.
STAGE_ORDER: list[str] = ["resolve", "audio", "probe", "asr", "index", "match", "frame"]


def _stage_base(stage: str) -> float:
    """Cumulative weight of all stages BEFORE `stage`.

    WHY: overall progress for a stage that is 50% done is
    (everything before it) + (half of its own weight).

    USED BY: `_OverallProgress.__call__`.
    """
    total = 0.0
    for name in STAGE_ORDER:
        if name == stage:
            return total
        total += STAGE_WEIGHTS.get(name, 0.0)
    return total


class _OverallProgress:
    """Wraps a caller's callback to add an overall-percentage estimate.

    WHY THIS EXISTS
        Core modules only know their OWN progress -- asr.py can say "60% of the
        audio transcribed" but has no idea it is one stage of six. A progress
        bar in a browser needs a single number that only ever moves forward.

        This adapter converts (stage, stage_percent) into an overall percent
        using STAGE_WEIGHTS, and clamps it monotonically so a cached stage
        completing instantly never causes the bar to jump backwards.

    USED BY: `find_dialogue`, wrapping whatever callback the caller supplied.
    """

    def __init__(self, inner: Optional[ProgressCallback]) -> None:
        """Wrap `inner`, or fall back to the stderr sink when it is None."""
        # None means "the caller wants the default sink", not "discard". Pass
        # progress.null_progress explicitly to silence output instead.
        self._inner: ProgressCallback = inner if inner is not None else stderr_progress
        self._highest = 0.0
        # Stage labels seen so far, in arrival order -- reported to the UI so it
        # can show which stages have already run.
        self.seen: list[str] = []

    def __call__(self, stage: str, percent: Optional[float], message: str) -> None:
        """Receive one per-stage event and forward it with an overall percent.

        Callable rather than a method so it can be passed anywhere a plain
        ProgressCallback is expected -- the core modules never know it is not a
        plain function.
        """
        if stage not in self.seen:
            self.seen.append(stage)

        # percent=None means "this stage cannot measure itself", NOT "finished".
        # Treating it as 100 would pin the bar to the END of the stage on the
        # stage's first message, and the monotonic clamp below would then block
        # every real reading that followed. Treat it as the stage's start.
        within = 0.0 if percent is None else max(0.0, min(100.0, percent))
        overall = _stage_base(stage) + STAGE_WEIGHTS.get(stage, 0.0) * within / 100.0

        # Never go backwards: a bar that retreats reads as a bug to a user.
        self._highest = max(self._highest, min(overall, 99.0))
        self._inner(stage, round(self._highest, 1), message)

    def complete(self, message: str = "done") -> None:
        """Emit a final 100% event.

        USED BY: `find_dialogue` once the result is fully assembled, so the web
        UI can close its progress bar cleanly rather than leaving it at 99%.
        """
        self._highest = 100.0
        self._inner("done", 100.0, message)


@dataclass
class DialogueResult:
    """The complete answer to one query, as data rather than printed text.

    WHY A DATACLASS: the CLI needs to format it, the API needs to serialise it,
    and a future caller may want to inspect individual fields. Returning a
    string would force every consumer to parse it back apart.

    USED BY: app/cli.py (via print_result) and app/api.py (via to_dict).
    """

    status: str                                # found | not_found
    media_key: str
    title: str
    source_url: str
    query: str

    # --- populated only when status == "found" ---
    timestamp: Optional[str] = None            # HH:MM:SS.sss, the required format
    timestamp_seconds: Optional[float] = None
    frame_number: Optional[int] = None         # None for variable-frame-rate sources
    frame_pts: Optional[float] = None          # presentation time of that exact frame
    matched_text: Optional[str] = None         # original transcript words, not normalized
    image_path: Optional[str] = None

    score: Optional[float] = None              # rapidfuzz partial_ratio, 0-100
    band: str = config.BAND_NO_MATCH           # confident | ambiguous | no_match
    context: Optional[str] = None              # surrounding words, for judging a hit
    frame_note: Optional[str] = None           # why frame_number is null, or a clamp warning

    other_occurrences: list[dict[str, Any]] = field(default_factory=list)
    near_misses: list[dict[str, Any]] = field(default_factory=list)
    cached_stages: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form.

        USED BY: `python -m app.cli --json` and every JSON response in app/api.py.
        """
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

    THE ONE ENTRY POINT for the whole system. Runs resolve -> audio -> probe ->
    index -> match -> frame, reusing every cached artifact it can, and returns
    structured data. Prints nothing.

    Args:
        url: A single video URL. Playlists, live streams and DRM are rejected
             by app/core/resolve.py before any work happens.
        text: The dialogue to find. Normalized with the same function used at
             index time, so typed punctuation and casing do not matter.
        force: Ignore every cache and recompute from scratch. Useful after
             changing the ASR model.
        progress_callback: Receives (stage, overall_percent, message). See
             app/progress.py for the contract.

    Returns:
        DialogueResult with status "found" or "not_found". A not_found result
        carries `near_misses` so the caller can show what the transcript
        actually says instead of fabricating an answer.

    Raises:
        Quest1Error subclasses for bad input, unsupported media, or a genuine
        processing failure. Callers are expected to catch these and present the
        message -- they are all written to be shown to a user verbatim.

    USED BY: app/cli.py and app/api.py.
    """
    started = time.time()
    progress = _OverallProgress(progress_callback)
    cached: list[str] = []

    # --- 1. Resolve -------------------------------------------------------- #
    # Cached per URL while the signed stream URLs remain valid, because a fresh
    # yt-dlp extraction costs seconds and would dominate a repeat query.
    media: ResolvedMedia
    media, resolve_hit = resolve_cached(
        url, force=force, check_ranges=False, progress_callback=progress,
    )
    if resolve_hit:
        cached.append("resolve")

    # --- 2. Audio + probe -------------------------------------------------- #
    # Existence is checked BEFORE calling ensure_*, because those functions
    # return the same value whether they built or reused the artifact; the only
    # way to report "this was cached" honestly is to look first.
    audio_existed = not force and paths.audio_path(media.media_key).exists()
    wav = ensure_audio(media, force=force, progress_callback=progress)
    if audio_existed:
        cached.append("audio")

    probe_existed = not force and paths.probe_path(media.media_key).exists()
    probe = ensure_probe(media, force=force, progress_callback=progress)
    if probe_existed:
        cached.append("probe")

    # --- 3. Transcript index ----------------------------------------------- #
    # The expensive stage, and the one this entire caching design exists to
    # skip on repeat queries.
    index = ensure_index(media.media_key, wav, force=force, progress_callback=progress)
    if index.from_cache:
        cached.append("index")

    # --- 4. Match ---------------------------------------------------------- #
    # Never cached: it takes milliseconds and depends entirely on the query.
    match: MatchResult = find_matches(index, text, progress_callback=progress)

    if not match.found:
        report(progress, STAGE, "no match above threshold; returning near misses")
        progress.complete("no match")
        return DialogueResult(
            status="not_found",
            media_key=media.media_key,
            title=media.title,
            source_url=media.source_url,
            query=text,
            band=match.band,
            near_misses=[o.to_dict() for o in match.near_misses],
            cached_stages=cached,
            elapsed_seconds=time.time() - started,
        )

    best = match.best
    assert best is not None  # guaranteed by match.found being True

    # --- 5. Frame ---------------------------------------------------------- #
    # best.start_time is when the matched word BEGINS being spoken, which is the
    # honest answer to "where is this line spoken". Nudging into the word would
    # look better on fast speech but would no longer be the frame the dialogue
    # starts on.
    frame_path = paths.frames_dir(media.media_key) / frame_mod.frame_filename(best.start_time)
    frame_existed = not force and frame_path.exists()
    frame_result = frame_mod.extract_frame(
        media, best.start_time, probe, force=force, progress_callback=progress,
    )
    if frame_existed:
        cached.append("frame")

    progress.complete(f"found at {format_timestamp(best.start_time)}")
    return DialogueResult(
        status="found",
        media_key=media.media_key,
        title=media.title,
        source_url=media.source_url,
        query=text,
        timestamp=format_timestamp(best.start_time),
        timestamp_seconds=best.start_time,
        frame_number=frame_result.frame_number,
        frame_pts=frame_result.frame_pts,
        matched_text=best.matched_text,
        image_path=frame_result.path,
        score=best.score,
        band=match.band,
        context=best.context,
        frame_note=frame_result.note,
        # Every other place the line was said, so an ambiguous result can be
        # judged by a human rather than silently resolved by us.
        other_occurrences=[o.to_dict() for o in match.occurrences if o is not best],
        near_misses=[],
        cached_stages=cached,
        elapsed_seconds=time.time() - started,
    )
