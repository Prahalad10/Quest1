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
    """One place in the transcript where the target text appears."""

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
        return asdict(self)


@dataclass
class MatchResult:
    """The full answer to one query, including why it might be uncertain."""

    media_key: str
    query: str
    normalized_query: str
    band: str                              # confident | ambiguous | no_match
    occurrences: list[Occurrence] = field(default_factory=list)   # sorted by time
    near_misses: list[Occurrence] = field(default_factory=list)   # only when no_match
    threshold: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.occurrences)

    @property
    def best(self) -> Optional[Occurrence]:
        """Highest-scoring occurrence; ties broken by earliest in the video."""
        if not self.occurrences:
            return None
        return max(self.occurrences, key=lambda o: (o.score, -o.start_time))

    def to_dict(self) -> dict[str, Any]:
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
    try:
        from rapidfuzz import fuzz
    except ImportError as exc:  # pragma: no cover - environment problem
        raise Quest1Error(
            "rapidfuzz is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return fuzz


def _build_occurrence(index: TranscriptIndex, score: float, start: int, end: int) -> Optional[Occurrence]:
    """Turn a character span into a fully resolved occurrence, or None if unusable."""
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
    """Mask-and-repeat search returning every span scoring >= threshold."""
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

    A high score alone is not enough: if a second occurrence scores within
    `margin` of the best, we genuinely cannot tell which one the caller meant,
    so the answer is ambiguous even though the match itself is strong.
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
    """Locate `query` within `index`.

    Returns every occurrence at or above threshold sorted by time, plus a
    confidence band. When nothing matches, `near_misses` holds the closest few
    spans so the caller can show what the transcript actually says instead of
    fabricating an answer.
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
    """Convenience wrapper: load the cached index for `media_key`, then match."""
    paths.validate_media_key(media_key)
    index = load_index(media_key)
    if index is None:
        raise InvalidInputError(
            f"No valid transcript index for {media_key!r}. "
            f"Run: python -m app.core.index {media_key}"
        )
    return find_matches(index, query, progress_callback=progress_callback, **kwargs)


def format_timestamp(seconds: float) -> str:
    """HH:MM:SS.sss -- the output format the CLI must produce."""
    if seconds < 0:
        seconds = 0.0
    hours, remainder = divmod(float(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _print_occurrence(label: str, occ: Occurrence) -> None:
    print(f"  {label}")
    print(f"    score      : {occ.score}")
    print(f"    time       : {format_timestamp(occ.start_time)} -> "
          f"{format_timestamp(occ.end_time)}  ({occ.start_time:.2f}s)")
    print(f"    words      : {occ.first_word}..{occ.last_word}")
    print(f"    matched    : {occ.matched_text!r}")
    print(f"    context    : ...{occ.context}...")


def main(argv: Optional[list[str]] = None) -> int:
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
