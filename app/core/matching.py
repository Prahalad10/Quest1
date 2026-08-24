"""Find where a line of dialogue occurs in a transcript index.

This is the QUERYING half of the system: cheap, per-text, and it never touches
ASR. It reads the flat normalized string built by index.py, fuzzy-matches the
target against it, and maps the winning character span back to word indices --
and therefore to timestamps.

Deliberately source-agnostic: it matches against a `TranscriptIndex` and knows
nothing about where the text came from. A future visual/OCR path can produce the
same structure and reuse this module unchanged.

Why fuzzy: ASR output rarely matches typed dialogue exactly. "we're gonna need a
bigger boat" may be transcribed "we are going to need a bigger boat", so an exact
substring search would find nothing while a human would call it a clear hit.

Multiple occurrences use mask-and-repeat: find the best span, blank it out with
NULs (which preserves every character offset), search again, and stop when the
next best drops below threshold.

Run directly:
    python -m app.core.matching <media_key> "<dialogue text>"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from app import config, paths
from app.core.index import TranscriptIndex, load_index
from app.core.normalize import normalize_text
from app.errors import InvalidInputError, Quest1Error
from app.progress import ProgressCallback, report

STAGE = "match"

# Masking character: cannot appear in normalized text, and keeps offsets stable.
_MASK = "\x00"


@dataclass
class Occurrence:
    """One place in the transcript where the target text appears.

    Carries both the machine view (score, char span, word indices) and the human
    view (matched_text, context) because an ambiguous result has to be JUDGED by
    a person, and a bare timestamp gives them nothing to judge with.

    USED BY: MatchResult, app/service.py (turns the best one into a frame), and
    the web UI (renders other occurrences and near misses).
    """

    score: float
    char_start: int
    char_end: int
    first_word: int
    last_word: int
    start_time: float
    end_time: float
    matched_text: str      # the original, un-normalized words -- what we show the user
    normalized_match: str  # the exact span that matched, for debugging
    context: str           # surrounding words, so an ambiguous hit can be judged

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, used wherever this is persisted or returned."""
        return asdict(self)


@dataclass
class MatchResult:
    """The full answer to one query, including why it might be uncertain.

    Deliberately returns EVERY occurrence above threshold, not just the winner.
    A line said three times has three legitimate answers, and only the user can
    say which one they meant -- silently picking one and hiding the rest would
    make a guess look like a fact.

    USED BY: app/service.py, which promotes `best` into the final answer and
    passes the rest through as other_occurrences.
    """

    media_key: str
    query: str
    normalized_query: str
    band: str                              # confident | ambiguous | no_match
    occurrences: list[Occurrence] = field(default_factory=list)   # sorted by time
    near_misses: list[Occurrence] = field(default_factory=list)   # only when no_match
    threshold: float = 0.0

    @property
    def found(self) -> bool:
        """True when at least one occurrence cleared the threshold.

        USED BY: app/service.py to choose between the found and not_found paths.
        """
        return bool(self.occurrences)

    @property
    def best(self) -> Optional[Occurrence]:
        """Highest-scoring occurrence; ties broken by EARLIEST in the video.

        WHY EARLIEST WINS A TIE: with several identical matches something has to
        decide, and "the first time it is said" is the least surprising rule.
        The alternatives are all arbitrary, and the losing occurrences are
        returned anyway so nothing is hidden.

        USED BY: app/service.py (selects the timestamp to extract a frame from)
        and the CLI (marks it in the printed list).
        """
        if not self.occurrences:
            return None
        return max(self.occurrences, key=lambda o: (o.score, -o.start_time))

    def to_dict(self) -> dict[str, Any]:
        """JSON form. USED BY: `python -m app.core.matching --json`."""
        return {
            "media_key": self.media_key,
            "query": self.query,
            "normalized_query": self.normalized_query,
            "band": self.band,
            "found": self.found,
            "threshold": self.threshold,
            "best": self.best.to_dict() if self.best else None,
            "occurrences": [o.to_dict() for o in self.occurrences],
            "near_misses": [o.to_dict() for o in self.near_misses],
        }


def _require_rapidfuzz():
    """Import rapidfuzz lazily, with an actionable message if it is absent.

    WHY LAZY: keeps module import cheap for callers that only need the dataclasses
    or format_timestamp, and turns a missing dependency into an instruction
    rather than an ImportError traceback.

    USED BY: _search.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError as exc:  # pragma: no cover - environment problem
        raise Quest1Error(
            "rapidfuzz is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return fuzz


def _build_occurrence(index: TranscriptIndex, score: float, start: int, end: int) -> Optional[Occurrence]:
    """Turn a character span into a fully resolved occurrence, or None if unusable.

    Where character offsets become TIMESTAMPS: the span is mapped to a word range
    via the index offset array, and the word range to a time range.

    Returns None (rather than raising) for a span made entirely of separators,
    which cannot name a word. _search skips those instead of reporting a match
    with no timestamp.

    USED BY: _search.
    """
    try:
        first_word, last_word = index.span_to_word_range(start, end)
    except Quest1Error:
        # A span consisting only of separators cannot name a word. Skip it rather
        # than reporting a match with no timestamp.
        return None

    start_time, end_time = index.word_time_span(first_word, last_word)
    return Occurrence(
        score=round(float(score), 2),
        char_start=start,
        char_end=end,
        first_word=first_word,
        last_word=last_word,
        start_time=start_time,
        end_time=end_time,
        matched_text=" ".join(w["word"] for w in index.words[first_word:last_word + 1]),
        normalized_match=index.normalized_text[start:end].strip(),
        context=index.context_text(first_word, last_word),
    )


def _search(
    index: TranscriptIndex,
    needle: str,
    *,
    threshold: float,
    limit: int,
) -> list[Occurrence]:
    """Mask-and-repeat search returning every span scoring >= threshold.

    HOW IT FINDS ALL OCCURRENCES: rapidfuzz partial_ratio_alignment returns only
    the single BEST alignment. To find the rest, the winning span is overwritten
    with NUL bytes and the search repeated. NUL is used because it cannot appear
    in normalized text and, crucially, preserves the length of the string -- so
    every character offset still refers to the same place in the original.

    The zero-width guard is defensive: a zero-length span would never be masked
    and the loop would spin forever.

    USED BY: find_matches, for both real matches and near misses (the only
    difference is the threshold).
    """
    fuzz = _require_rapidfuzz()
    haystack = index.normalized_text
    found: list[Occurrence] = []

    while len(found) < limit:
        alignment = fuzz.partial_ratio_alignment(needle, haystack, score_cutoff=threshold)
        if alignment is None:
            break

        start, end = alignment.dest_start, alignment.dest_end
        if end <= start:
            break  # defensive: a zero-width span would loop forever

        occurrence = _build_occurrence(index, alignment.score, start, end)
        if occurrence is not None:
            found.append(occurrence)

        # Blank the span so the next iteration finds the *next* best match.
        # NUL keeps every later offset identical to the original string.
        haystack = haystack[:start] + (_MASK * (end - start)) + haystack[end:]

    return found


def classify(occurrences: list[Occurrence], *, confident_threshold: float, margin: float) -> str:
    """Decide the confidence band for a set of occurrences.

    THE ARBITRATION RULE. A high score alone is not enough to be confident: if a
    second occurrence scores within `margin` of the best, we genuinely cannot
    tell which one the caller meant, so the answer is downgraded to ambiguous
    even though the match itself is perfect.

    This is what stops the system presenting a coin-flip as a fact. The three
    bands come from config so they can be tuned without touching this logic.

    Deliberately source-agnostic: it takes a list of occurrences and nothing
    else, so a future visual/OCR path could reuse it unchanged.

    USED BY: find_matches.
    """
    if not occurrences:
        return config.BAND_NO_MATCH

    ranked = sorted(occurrences, key=lambda o: o.score, reverse=True)
    best = ranked[0]
    if best.score < confident_threshold:
        return config.BAND_AMBIGUOUS
    if len(ranked) > 1 and (best.score - ranked[1].score) <= margin:
        return config.BAND_AMBIGUOUS
    return config.BAND_CONFIDENT


def find_matches(
    index: TranscriptIndex,
    query: str,
    *,
    threshold: Optional[float] = None,
    confident_threshold: Optional[float] = None,
    margin: Optional[float] = None,
    limit: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> MatchResult:
    """Locate `query` within `index`. THE QUERY ENTRY POINT.

    Cheap by design -- milliseconds against an already-built index -- which is
    what makes a second search on the same video fast.

    The query is normalized with the SAME function used at index time, so typed
    punctuation, casing and curly quotes are irrelevant to whether it matches.

    Returns every occurrence at or above threshold sorted by TIME (not score),
    plus a confidence band. When nothing matches, `near_misses` holds the closest
    few spans so the caller can show what the transcript actually says instead of
    fabricating an answer.

    Raises InvalidInputError for empty text, text that normalizes to nothing, or
    a query too short to produce a meaningful timestamp.

    USED BY: app/service.py (stage 5) and find_in_media below.
    """
    if not isinstance(query, str) or not query.strip():
        raise InvalidInputError("Dialogue text is required and must be a non-empty string.")

    threshold = config.MATCH_THRESHOLD if threshold is None else threshold
    confident_threshold = (
        config.CONFIDENT_THRESHOLD if confident_threshold is None else confident_threshold
    )
    margin = config.AMBIGUITY_MARGIN if margin is None else margin
    limit = config.MAX_OCCURRENCES if limit is None else limit

    # Same normalization as index time -- this is the whole point of the shared function.
    needle = normalize_text(query)
    if not needle:
        raise InvalidInputError(
            f"Dialogue text {query!r} contains no matchable characters after normalization."
        )
    if len(needle) < config.MIN_QUERY_CHARS:
        raise InvalidInputError(
            f"Dialogue text {query!r} normalizes to {needle!r}, which is shorter than the "
            f"{config.MIN_QUERY_CHARS}-character minimum. Such a short query would match "
            f"almost anywhere and its timestamp would be meaningless."
        )

    report(progress_callback, STAGE, f"searching {index.word_count} words for {needle!r}")
    occurrences = _search(index, needle, threshold=threshold, limit=limit)
    band = classify(occurrences, confident_threshold=confident_threshold, margin=margin)

    near_misses: list[Occurrence] = []
    if not occurrences:
        near_misses = _search(
            index, needle,
            threshold=config.NEAR_MISS_THRESHOLD,
            limit=config.NEAR_MISS_COUNT,
        )
        near_misses.sort(key=lambda o: o.score, reverse=True)
        report(
            progress_callback, STAGE,
            f"no match above {threshold}; {len(near_misses)} near miss(es)",
        )
    else:
        occurrences.sort(key=lambda o: o.start_time)   # chronological, per spec
        report(
            progress_callback, STAGE,
            f"{len(occurrences)} occurrence(s), band={band}, "
            f"best score={max(o.score for o in occurrences)}",
        )

    return MatchResult(
        media_key=index.media_key,
        query=query,
        normalized_query=needle,
        band=band,
        occurrences=occurrences,
        near_misses=near_misses,
        threshold=threshold,
    )


def find_in_media(
    media_key: str,
    query: str,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    **kwargs: Any,
) -> MatchResult:
    """Convenience wrapper: load the cached index for `media_key`, then match.

    Deliberately does NOT build an index if one is missing -- ASR takes minutes
    and should never be triggered as a side effect of a query. It says how to
    build it instead.

    USED BY: the __main__ block below. app/service.py calls find_matches directly
    because it already holds the index.
    """
    paths.validate_media_key(media_key)
    index = load_index(media_key)
    if index is None:
        raise InvalidInputError(
            f"No valid transcript index for {media_key!r}. "
            f"Run: python -m app.core.index {media_key}"
        )
    return find_matches(index, query, progress_callback=progress_callback, **kwargs)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.sss -- the required output format.

    Millisecond precision is kept because at 30fps a single frame is 33ms; a
    coarser timestamp could not distinguish adjacent frames.

    USED BY: app/service.py (the timestamp field), app/cli.py, and the web UI
    via the JSON response.
    """
    if seconds < 0:
        seconds = 0.0
    hours, remainder = divmod(float(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _print_occurrence(label: str, occ: Occurrence) -> None:
    """Print one occurrence as an indented block.

    USED BY: main(), for matches and near misses alike so both read the same.
    """
    print(f"  {label}")
    print(f"    score      : {occ.score}")
    print(f"    time       : {format_timestamp(occ.start_time)} -> "
          f"{format_timestamp(occ.end_time)}  ({occ.start_time:.2f}s)")
    print(f"    words      : {occ.first_word}..{occ.last_word}")
    print(f"    matched    : {occ.matched_text!r}")
    print(f"    context    : ...{occ.context}...")


def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry point: query an already-built index and print matches.

    WHY: lets matching be tuned in isolation. Because it needs no network and no
    model, --threshold can be swept in a fraction of a second to see how the
    confidence bands behave on real transcript text.

    Exit code 1 on no-match, so a shell can distinguish "absent" from "broken".

    USED BY: `python -m app.core.matching <media_key> "<text>"`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.core.matching",
        description="Find a line of dialogue in a cached transcript index.",
    )
    parser.add_argument("media_key", help="media_key (see app.core.resolve)")
    parser.add_argument("text", help="Dialogue text to locate")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Match threshold 0-100 (default {config.MATCH_THRESHOLD})")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)

    try:
        result = find_in_media(args.media_key, args.text, threshold=args.threshold)
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if result.found else 1

    print()
    print(f"query      : {result.query!r}")
    print(f"normalized : {result.normalized_query!r}")
    print(f"band       : {result.band}")
    print(f"threshold  : {result.threshold}")
    print(f"occurrences: {len(result.occurrences)}")
    print()

    if result.found:
        best = result.best
        for i, occ in enumerate(result.occurrences):
            marker = "  <-- best" if occ is best else ""
            _print_occurrence(f"[{i}] at {format_timestamp(occ.start_time)}{marker}", occ)
            print()
        return 0

    print("  NO MATCH above threshold.")
    print()
    if result.near_misses:
        print(f"  closest {len(result.near_misses)} span(s) in the transcript:")
        print()
        for i, occ in enumerate(result.near_misses):
            _print_occurrence(f"[near miss {i}] score {occ.score}", occ)
            print()
    else:
        print(f"  Nothing in the transcript scored above "
              f"{config.NEAR_MISS_THRESHOLD}, so there is no useful suggestion.")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
