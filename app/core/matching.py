"""Fuzzy search over a built transcript index -- the QUERY half.

Milliseconds against an existing index, which is what makes a repeat search on
the same video fast. The query is normalized with the SAME function used at
index time, so typed punctuation and casing are irrelevant.

Returns EVERY occurrence above threshold, not just the winner: when a line is
said more than once only a person can say which one they meant, and silently
picking one would make a guess look like a fact.

    python -m app.core.matching <media_key> "<dialogue>"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from app import config, paths
from app.core.index import TranscriptIndex, load_index
from app.core.normalize import normalize_text
from app.errors import InvalidInputError, Quest1Error
from app.progress import ProgressCallback, report

STAGE = "match"

# NUL cannot appear in normalized text and preserves string length, so masking
# with it keeps every later character offset valid. See _search.
_MASK = "\x00"


@dataclass
class Occurrence:
    """One place the target text appears.

    Carries the machine view (score, spans) and the human view (matched_text,
    context) because an ambiguous result has to be judged by a person, and a
    bare timestamp gives them nothing to judge with.
    """

    score: float
    char_start: int
    char_end: int
    first_word: int
    last_word: int
    start_time: float
    end_time: float
    matched_text: str      # original, un-normalized words -- what the user sees
    normalized_match: str  # the exact span that matched, for debugging
    context: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    """The full answer to one query, including why it might be uncertain."""

    media_key: str
    query: str
    normalized_query: str
    band: str                                                    # confident|ambiguous|no_match
    occurrences: list[Occurrence] = field(default_factory=list)  # sorted by time
    near_misses: list[Occurrence] = field(default_factory=list)  # only when no_match
    threshold: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.occurrences)

    @property
    def best(self) -> Optional[Occurrence]:
        """Highest score; ties broken by EARLIEST in the video.

        Something has to decide, and "the first time it is said" is the least
        surprising rule. The losing occurrences are returned anyway.
        """
        if not self.occurrences:
            return None
        return sorted(self.occurrences, key=lambda o: (-o.score, o.start_time))[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_key": self.media_key,
            "query": self.query,
            "normalized_query": self.normalized_query,
            "band": self.band,
            "threshold": self.threshold,
            "occurrences": [o.to_dict() for o in self.occurrences],
            "near_misses": [o.to_dict() for o in self.near_misses],
        }


def _require_rapidfuzz():
    try:
        from rapidfuzz import fuzz
    except ImportError as exc:  # pragma: no cover - environment problem
        raise Quest1Error("rapidfuzz is not installed. Run: pip install -r requirements.txt") from exc
    return fuzz


def _build_occurrence(
    index: TranscriptIndex, score: float, start: int, end: int
) -> Optional[Occurrence]:
    """Where character offsets become TIMESTAMPS."""
    try:
        first_word, last_word = index.span_to_word_range(start, end)
    except Quest1Error:
        # A span of only separators cannot name a word; skip it rather than
        # report a match with no timestamp.
        return None
    start_time, end_time = index.word_time_span(first_word, last_word)
    return Occurrence(
        score=round(float(score), 2),
        char_start=start, char_end=end,
        first_word=first_word, last_word=last_word,
        start_time=start_time, end_time=end_time,
        matched_text=" ".join(w["word"] for w in index.words[first_word:last_word + 1]),
        normalized_match=index.normalized_text[start:end].strip(),
        context=index.context_text(first_word, last_word),
    )


def _search(
    index: TranscriptIndex, needle: str, *, threshold: float, limit: int
) -> list[Occurrence]:
    """Every span scoring >= threshold, by mask-and-repeat.

    partial_ratio_alignment returns only the single BEST alignment, so to find
    the rest the winning span is overwritten with NUL and the search repeated.
    NUL preserves the string's length, so every offset still refers to the same
    place in the original.
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
            break  # defensive: a zero-width span would never be masked and would loop
        occurrence = _build_occurrence(index, alignment.score, start, end)
        if occurrence is not None:
            found.append(occurrence)
        haystack = haystack[:start] + (_MASK * (end - start)) + haystack[end:]

    return found


def classify(occurrences: list[Occurrence], *, confident_threshold: float, margin: float) -> str:
    """Confidence band.

    A perfect match is still ambiguous if a runner-up scores within `margin`:
    we genuinely cannot tell which one the caller meant.
    """
    if not occurrences:
        return config.BAND_NO_MATCH
    ranked = sorted(occurrences, key=lambda o: o.score, reverse=True)
    if ranked[0].score < confident_threshold:
        return config.BAND_AMBIGUOUS
    if len(ranked) > 1 and (ranked[0].score - ranked[1].score) <= margin:
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

    When nothing matches, near_misses holds the closest few spans so the caller
    can show what the transcript actually says instead of fabricating an answer.
    """
    if not isinstance(query, str) or not query.strip():
        raise InvalidInputError("Dialogue text is required and must be a non-empty string.")

    threshold = config.MATCH_THRESHOLD if threshold is None else threshold
    confident_threshold = (config.CONFIDENT_THRESHOLD if confident_threshold is None
                           else confident_threshold)
    margin = config.AMBIGUITY_MARGIN if margin is None else margin
    limit = config.MAX_OCCURRENCES if limit is None else limit

    needle = normalize_text(query)  # same folding as index time -- the whole point
    if not needle:
        raise InvalidInputError(
            f"Dialogue text {query!r} contains no matchable characters after normalization."
        )
    if len(needle) < config.MIN_QUERY_CHARS:
        raise InvalidInputError(
            f"Dialogue text {query!r} normalizes to {needle!r}, shorter than the "
            f"{config.MIN_QUERY_CHARS}-character minimum. Such a short query would match "
            f"almost anywhere and its timestamp would be meaningless."
        )

    report(progress_callback, STAGE, f"searching {index.word_count} words for {needle!r}")
    occurrences = _search(index, needle, threshold=threshold, limit=limit)
    band = classify(occurrences, confident_threshold=confident_threshold, margin=margin)
    near_misses: list[Occurrence] = []

    if not occurrences:
        near_misses = _search(index, needle, threshold=config.NEAR_MISS_THRESHOLD,
                              limit=config.NEAR_MISS_COUNT)
        near_misses.sort(key=lambda o: o.score, reverse=True)
        report(progress_callback, STAGE,
               f"no match above {threshold}; {len(near_misses)} near miss(es)")
    else:
        occurrences.sort(key=lambda o: o.start_time)
        report(progress_callback, STAGE,
               f"{len(occurrences)} occurrence(s), band={band}, "
               f"best score={max(o.score for o in occurrences)}")

    return MatchResult(media_key=index.media_key, query=query, normalized_query=needle,
                       band=band, occurrences=occurrences, near_misses=near_misses,
                       threshold=threshold)


def find_in_media(
    media_key: str,
    query: str,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    **kwargs: Any,
) -> MatchResult:
    """Load the cached index for `media_key`, then match.

    Deliberately does NOT build an index if one is missing: ASR takes minutes
    and must never run as a side effect of a query.
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
    """HH:MM:SS.sss. Milliseconds matter: at 30fps a frame is 33ms."""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.core.matching")
    parser.add_argument("media_key")
    parser.add_argument("text")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)

    try:
        result = find_in_media(args.media_key, args.text, threshold=args.threshold)
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    print(f"\nquery      : {result.query!r} -> {result.normalized_query!r}")
    print(f"band       : {result.band}")
    print(f"occurrences: {len(result.occurrences)}\n")
    for occ in result.occurrences:
        marker = " <- best" if occ is result.best else ""
        print(f"  {format_timestamp(occ.start_time)}  score {occ.score:>6}  "
              f"{occ.matched_text!r}{marker}")
        print(f"      ...{occ.context}...")
    for occ in result.near_misses:
        print(f"  near miss  {format_timestamp(occ.start_time)}  score {occ.score:>6}  "
              f"{occ.matched_text!r}")
    print()
    return 0 if result.found else 1


if __name__ == "__main__":
    sys.exit(main())
