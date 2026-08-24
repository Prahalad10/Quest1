"""Build, persist, and load the per-video transcript index.

This is the INDEXING half of the system -- expensive, run once per video. The
QUERYING half (matching.py) only ever reads what this produces, which is what
lets a second search on the same video skip ASR entirely.

Artifacts under data/{media_key}/:

    transcript.json  segments, words, normalized_text, char_to_word
    index.json       manifest -- INDEX_VERSION, model, counts, timings

The two central structures:

    normalized_text   every word normalized and joined by single spaces, e.g.
                      "alright so here we are in front of the elephants"
    char_to_word      one entry per character of normalized_text, giving the
                      index into words[] that the character belongs to. The
                      space separators map to the word on their LEFT, so callers
                      should trim a span before mapping it (span_to_word_range
                      does this for you).

Run directly:
    python -m app.core.index <media_key> --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app import config, paths
from app.core.asr import Segment, Transcription, Word, transcribe
from app.core.normalize import normalize_text
from app.errors import InvalidInputError, Quest1Error
from app.progress import ProgressCallback, report

STAGE = "index"


class TranscriptIndexError(Quest1Error):
    """The transcript index is missing, corrupt, or internally inconsistent.

    Note the asymmetry in how this module reacts to problems: a STALE index
    (INDEX_VERSION mismatch) is not an error -- load_index returns None and the
    caller rebuilds. A CORRUPT index raises, because silently rebuilding over
    corruption would hide a real problem such as a failing disk.
    """


@dataclass
class TranscriptIndex:
    """Everything querying needs, with no ASR dependency.

    THE CENTRAL DATA STRUCTURE of the project. Built once per video (expensive),
    then read by every subsequent query (cheap). Because it carries the flat
    normalized text AND the char->word offset array, a query never needs the
    model, the audio, or the network.

    USED BY: matching.py (searches normalized_text, maps spans back through
    char_to_word) and app/service.py (holds it between stages).
    """

    media_key: str
    index_version: int
    model: str
    language: Optional[str]
    language_probability: Optional[float]
    duration: float
    segments: list[dict[str, Any]]
    words: list[dict[str, Any]]
    normalized_text: str
    char_to_word: list[int]
    created_at: float
    asr_elapsed_seconds: float = 0.0
    # True when loaded from disk rather than freshly built -- the CLI reports it.
    from_cache: bool = field(default=False, compare=False)

    @property
    def word_count(self) -> int:
        """Number of usable words. USED BY: progress messages and the manifest."""
        return len(self.words)

    def word_time_span(self, first: int, last: int) -> tuple[float, float]:
        """Start of word `first` to end of word `last`, inclusive.

        Indices are clamped rather than raising, and a reversed range is swapped,
        because callers derive them from fuzzy match spans where an off-by-one at
        the very edge of the transcript is possible and harmless.

        USED BY: matching._build_occurrence, to turn a word range into the
        timestamps that ultimately select the frame.
        """
        if not self.words:
            raise TranscriptIndexError("Index contains no words.")
        first = max(0, min(first, len(self.words) - 1))
        last = max(0, min(last, len(self.words) - 1))
        if last < first:
            first, last = last, first
        return float(self.words[first]["start"]), float(self.words[last]["end"])

    def span_to_word_range(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) char span of normalized_text to a word index range.

        THE BRIDGE between fuzzy matching (which works on characters) and
        timestamps (which exist per word). rapidfuzz returns character offsets;
        this converts them into the word indices whose times we need.

        Whitespace at either edge is trimmed FIRST. Separators are recorded as
        belonging to the word on their left, so a span that happens to begin on
        a space would otherwise pull in the entire preceding word and report a
        timestamp too early.

        USED BY: matching._build_occurrence.
        """
        text = self.normalized_text
        if not text:
            raise TranscriptIndexError("Index contains no normalized text.")
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            raise TranscriptIndexError(f"Character span [{start}, {end}) contains no words.")
        return self.char_to_word[start], self.char_to_word[end - 1]

    def context_text(self, first: int, last: int, padding: int = 6) -> str:
        """Original (un-normalized) words around a match, for display.

        WHY UN-NORMALIZED: the user should see what was actually said, with its
        punctuation and casing, not the lowercase stripped form used internally.

        WHY IT MATTERS: for an "ambiguous" result this is how a human decides
        whether the match is the line they meant.

        USED BY: matching._build_occurrence -> shown by the CLI and the web UI.
        """
        low = max(0, first - padding)
        high = min(len(self.words), last + 1 + padding)
        return " ".join(w["word"] for w in self.words[low:high])

    def to_transcript_dict(self) -> dict[str, Any]:
        """Serialise everything to the transcript.json schema.

        USED BY: _persist. The matching field list in load_index must be kept in
        step with this -- a change to either is a reason to bump INDEX_VERSION.
        """
        return {
            "media_key": self.media_key,
            "index_version": self.index_version,
            "model": self.model,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "created_at": self.created_at,
            "asr_elapsed_seconds": self.asr_elapsed_seconds,
            "segments": self.segments,
            "words": self.words,
            "normalized_text": self.normalized_text,
            "char_to_word": self.char_to_word,
        }


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def build_flat_text(words: list[Word]) -> tuple[str, list[int], list[Word]]:
    """Join normalized words into one string plus its char -> word index map.

    THE CORE INDEXING STEP. Produces the two structures every query depends on:

        flat          "alright so here we are one of the elephants ..."
        char_to_word  [0,0,0,0,0,0,0, 0, 1,1, 1, 2,2,2,2, ...]

    so that character i of the flat text can be traced back to the word that
    produced it, and from there to its timestamp.

    Words that normalize to nothing (pure punctuation) are DROPPED, which is why
    the function returns its own `kept` list: char_to_word indexes into that, not
    into the input. Returning the wrong one here would offset every timestamp.

    The separator between two words is attributed to the word on its LEFT; see
    TranscriptIndex.span_to_word_range for how that is handled at query time.

    USED BY: build_index.
    """
    kept: list[Word] = []
    pieces: list[str] = []
    char_to_word: list[int] = []

    for word in words:
        normalized = normalize_text(word.word)
        if not normalized:
            continue
        word_index = len(kept)
        if pieces:
            # The separator belongs to the word on its left.
            pieces.append(" ")
            char_to_word.append(word_index - 1)
        pieces.append(normalized)
        char_to_word.extend([word_index] * len(normalized))
        kept.append(word)

    flat = "".join(pieces)
    if len(flat) != len(char_to_word):
        raise TranscriptIndexError(
            f"Internal error: flat text is {len(flat)} chars but the offset array "
            f"has {len(char_to_word)} entries."
        )
    return flat, char_to_word, kept


def build_index(
    media_key: str,
    wav_path: Path | str,
    *,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> TranscriptIndex:
    """Run ASR and write transcript.json + index.json. Always recomputes.

    Call ensure_index instead unless you specifically want to force a rebuild --
    this function does no cache checking and will happily spend minutes
    re-transcribing a video that is already indexed.

    USED BY: ensure_index.
    """
    paths.validate_media_key(media_key)
    result: Transcription = transcribe(
        wav_path,
        model_name=model_name,
        language=language,
        progress_callback=progress_callback,
    )

    flat, char_to_word, kept_words = build_flat_text(result.words)
    if not flat:
        raise TranscriptIndexError(
            "Transcription produced words but none survived normalization -- the "
            "transcript appears to contain no matchable text."
        )

    index = TranscriptIndex(
        media_key=media_key,
        index_version=config.INDEX_VERSION,
        model=result.model,
        language=result.language,
        language_probability=result.language_probability,
        duration=result.duration,
        segments=[s.to_dict() for s in result.segments],
        words=[w.to_dict() for w in kept_words],
        normalized_text=flat,
        char_to_word=char_to_word,
        created_at=time.time(),
        asr_elapsed_seconds=result.elapsed_seconds,
    )
    _persist(index, progress_callback=progress_callback)
    return index


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a .part file then rename.

    WHY: load_index treats the presence of both files as proof of a usable
    index. A partially written transcript.json would be detected only as a JSON
    parse error much later, after the expensive ASR had already been discarded.

    USED BY: _persist.
    """
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def _persist(index: TranscriptIndex, *, progress_callback: Optional[ProgressCallback] = None) -> None:
    """Write transcript.json (the data) and index.json (the manifest).

    WHY TWO FILES: staleness is checked on every query, and transcript.json can
    be megabytes for a long video. Keeping INDEX_VERSION and the counts in a
    small separate manifest means the check costs a few hundred bytes instead of
    parsing the whole transcript.

    USED BY: build_index.
    """
    paths.ensure_media_dir(index.media_key)
    transcript_file = paths.transcript_path(index.media_key)
    index_file = paths.index_path(index.media_key)

    _write_json_atomic(transcript_file, index.to_transcript_dict())

    manifest = {
        "index_version": index.index_version,
        "media_key": index.media_key,
        "created_at": index.created_at,
        "model": index.model,
        "language": index.language,
        "language_probability": index.language_probability,
        "duration": index.duration,
        "word_count": index.word_count,
        "segment_count": len(index.segments),
        "normalized_char_count": len(index.normalized_text),
        "asr_elapsed_seconds": index.asr_elapsed_seconds,
        "artifacts": {
            "transcript": transcript_file.name,
            "audio": paths.audio_path(index.media_key).name,
            "probe": paths.probe_path(index.media_key).name,
        },
    }
    _write_json_atomic(index_file, manifest)
    report(
        progress_callback, STAGE,
        f"wrote {transcript_file.name} + {index_file.name} "
        f"({index.word_count} words, {len(index.normalized_text)} chars)",
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_index(media_key: str) -> Optional[TranscriptIndex]:
    """Return the cached index, or None if it is absent or stale.

    A stale index (INDEX_VERSION mismatch) is treated as ABSENT so the caller
    rebuilds -- that is the whole purpose of the version constant. A CORRUPT
    index RAISES, because silently rebuilding over corruption would hide a real
    problem and cost the user minutes of ASR without explanation.

    The final consistency check (len(normalized_text) == len(char_to_word)) is
    cheap insurance: if those ever disagreed, every mapped timestamp would be
    quietly wrong rather than obviously broken.

    USED BY: ensure_index, and matching.find_in_media for a query against an
    already-indexed video.
    """
    paths.validate_media_key(media_key)
    index_file = paths.index_path(media_key)
    transcript_file = paths.transcript_path(media_key)
    if not index_file.exists() or not transcript_file.exists():
        return None

    try:
        manifest = json.loads(index_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptIndexError(f"{index_file} is corrupt: {exc}. Delete it and re-run.") from exc

    if manifest.get("index_version") != config.INDEX_VERSION:
        return None

    try:
        data = json.loads(transcript_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptIndexError(f"{transcript_file} is corrupt: {exc}. Delete it and re-run.") from exc

    try:
        index = TranscriptIndex(
            media_key=data["media_key"],
            index_version=data["index_version"],
            model=data["model"],
            language=data.get("language"),
            language_probability=data.get("language_probability"),
            duration=data.get("duration", 0.0),
            segments=data.get("segments", []),
            words=data["words"],
            normalized_text=data["normalized_text"],
            char_to_word=data["char_to_word"],
            created_at=data.get("created_at", 0.0),
            asr_elapsed_seconds=data.get("asr_elapsed_seconds", 0.0),
            from_cache=True,
        )
    except KeyError as exc:
        raise TranscriptIndexError(
            f"{transcript_file} is missing required field {exc}. Delete it and re-run."
        ) from exc

    if len(index.normalized_text) != len(index.char_to_word):
        raise TranscriptIndexError(
            f"{transcript_file} is inconsistent: {len(index.normalized_text)} chars vs "
            f"{len(index.char_to_word)} offsets. Delete it and re-run."
        )
    return index


def ensure_index(
    media_key: str,
    wav_path: Path | str,
    *,
    force: bool = False,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> TranscriptIndex:
    """Load the cached index, or build it. THE cache gate for the whole system.

    This one function is why a second search on the same video takes a second
    instead of minutes: it is the only place ASR is triggered, and it triggers it
    only when there is no valid index on disk.

    Sets from_cache on the returned index so callers can report honestly which
    stages were reused.

    USED BY: app/service.py (stage 4).
    """
    if not force:
        cached = load_index(media_key)
        if cached is not None:
            report(
                progress_callback, STAGE,
                f"cache hit: {cached.word_count} words, model={cached.model} "
                f"(index v{cached.index_version})",
            )
            return cached
        report(progress_callback, STAGE, "no valid index on disk, running ASR")

    return build_index(
        media_key,
        wav_path,
        model_name=model_name,
        language=language,
        progress_callback=progress_callback,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry point: build or inspect a media_key transcript index.

    WHY: lets you confirm what was transcribed and where each word sits in the
    flat text, without running a query. The CHARS column shows each word span
    in normalized_text, which is the mapping matching.py relies on.

    USED BY: `python -m app.core.index <media_key> [--text] [--force]`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.core.index",
        description="Build or inspect the transcript index for a media_key.",
    )
    parser.add_argument("media_key", help="media_key (see app.core.resolve)")
    parser.add_argument("--limit", type=int, default=20, help="How many words to print")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    parser.add_argument("--model", default=None, help=f"Model (default {config.ASR_MODEL})")
    parser.add_argument("--language", default=None, help="Force a language code, e.g. en")
    parser.add_argument("--text", action="store_true", help="Print the full normalized text")
    args = parser.parse_args(argv)

    try:
        wav = paths.audio_path(args.media_key)
        if not wav.exists():
            raise InvalidInputError(
                f"No audio at {wav}. Run: python -m app.core.audio <video_url>"
            )
        index = ensure_index(
            args.media_key,
            wav,
            force=args.force,
            model_name=args.model,
            language=args.language,
        )
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"media_key     : {index.media_key}")
    print(f"index_version : {index.index_version}")
    print(f"source        : {'cache' if index.from_cache else 'freshly built'}")
    print(f"model         : {index.model}")
    print(f"language      : {index.language} (p={index.language_probability or 0:.2f})")
    print(f"duration      : {index.duration:.2f}s")
    print(f"segments      : {len(index.segments)}")
    print(f"words         : {index.word_count}")
    print(f"normalized    : {len(index.normalized_text)} chars")
    print(f"asr elapsed   : {index.asr_elapsed_seconds:.1f}s")

    if args.text:
        print()
        print("normalized_text:")
        print(f"  {index.normalized_text}")

    print()
    shown = min(args.limit, index.word_count)
    print(f"first {shown} words:")
    print(f"  {'#':>4}  {'START':>8}  {'END':>8}  {'PROB':>5}  {'CHARS':>11}  WORD")
    for i, word in enumerate(index.words[:args.limit]):
        first_char = index.char_to_word.index(i) if i in index.char_to_word else -1
        span = f"{first_char}..{first_char + len(normalize_text(word['word'])) - 1}"
        print(
            f"  {i:>4}  {word['start']:>8.2f}  {word['end']:>8.2f}  "
            f"{word['probability']:>5.2f}  {span:>11}  {word['word']}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
