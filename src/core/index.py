"""Build, persist, and load the per-video transcript index.

The INDEXING half: expensive, once per video. Querying (matching.py) only ever
reads what this produces, which is what lets a repeat search skip ASR.

The two structures that make matching possible:

    normalized_text   every word normalized and joined by single spaces
    char_to_word      one entry per character, giving the word it belongs to

rapidfuzz returns character offsets; word timestamps live per word. This array
is the bridge. Separators are attributed to the word on their LEFT, so callers
must trim a span before mapping it -- span_to_word_range does that.

    python -m src.core.index <media_key> --limit 20
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

from src import config, paths
from src.core.asr import Segment, Transcription, Word, transcribe
from src.core.normalize import normalize_text
from src.errors import DialogueFrameError
from src.progress import ProgressCallback, report

STAGE = "index"

# Progress emitted BEFORE transcription must not use STAGE. service.py turns
# per-stage progress into one monotonic overall percentage, and "index" sits
# AFTER "asr" -- reporting the cache miss under it drove the bar to the index
# stage's offset and pinned it there for the whole of ASR.
STAGE_CHECK = "index_check"


class TranscriptIndexError(DialogueFrameError):
    """Index is corrupt or internally inconsistent.

    A STALE index (version mismatch) is not an error -- load_index returns None
    and the caller rebuilds. A CORRUPT one raises, because silently rebuilding
    over corruption would hide a real problem such as a failing disk.
    """


@dataclass
class TranscriptIndex:
    """Everything querying needs, with no ASR dependency."""

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
    from_cache: bool = field(default=False, compare=False)

    @property
    def word_count(self) -> int:
        return len(self.words)

    def word_time_span(self, first: int, last: int) -> tuple[float, float]:
        """Start of word `first` to end of word `last`.

        Indices are clamped rather than raising: callers derive them from fuzzy
        match spans where an off-by-one at the transcript edge is harmless.
        """
        if not self.words:
            raise TranscriptIndexError("Index contains no words.")
        first = max(0, min(first, len(self.words) - 1))
        last = max(0, min(last, len(self.words) - 1))
        if last < first:
            first, last = last, first
        return float(self.words[first]["start"]), float(self.words[last]["end"])

    def span_to_word_range(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) char span to a word index range.

        Whitespace at either edge is trimmed FIRST: separators belong to the
        word on their left, so a span beginning on a space would otherwise pull
        in the whole preceding word and report a timestamp too early.
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
        """Original words around a match, so a user can judge it in real prose."""
        low = max(0, first - padding)
        high = min(len(self.words), last + 1 + padding)
        return " ".join(w["word"] for w in self.words[low:high])

    def to_transcript_dict(self) -> dict[str, Any]:
        """The transcript.json schema. Changing it means bumping INDEX_VERSION."""
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


def build_flat_text(words: list[Word]) -> tuple[str, list[int], list[Word]]:
    """Join normalized words into one string plus its char -> word index map.

    Words that normalize to nothing (pure punctuation) are DROPPED, which is
    why this returns its own `kept` list: char_to_word indexes into that, not
    into the input. Returning the wrong one would offset every timestamp.
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
            pieces.append(" ")
            char_to_word.append(word_index - 1)  # separator belongs to the left word
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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """load_index treats both files existing as proof of a usable index, so a
    half-written transcript must never be visible."""
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def _persist(index: TranscriptIndex, *, progress_callback: Optional[ProgressCallback] = None) -> None:
    """Write transcript.json (data) and index.json (manifest).

    Two files because staleness is checked on every query and transcript.json
    can be megabytes; the check should cost a few hundred bytes, not a full parse.
    """
    paths.ensure_media_dir(index.media_key)
    transcript_file = paths.transcript_path(index.media_key)
    index_file = paths.index_path(index.media_key)

    _write_json_atomic(transcript_file, index.to_transcript_dict())
    _write_json_atomic(index_file, {
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
    })
    report(progress_callback, STAGE,
           f"wrote {transcript_file.name} + {index_file.name} "
           f"({index.word_count} words, {len(index.normalized_text)} chars)")


def build_index(
    media_key: str,
    wav_path: Path | str,
    *,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> TranscriptIndex:
    """Run ASR and write both artifacts. Always recomputes -- prefer ensure_index."""
    paths.validate_media_key(media_key)
    result: Transcription = transcribe(
        wav_path, model_name=model_name, language=language,
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


def load_index(media_key: str) -> Optional[TranscriptIndex]:
    """Cached index, or None if absent or stale.

    The final length check is cheap insurance: if text and offsets ever
    disagreed, every mapped timestamp would be quietly wrong rather than
    obviously broken.
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
        raise TranscriptIndexError(
            f"{transcript_file} is corrupt: {exc}. Delete it and re-run."
        ) from exc

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
    """Load the cached index or build it. THE cache gate for the whole system.

    This one function is why a second search on the same video takes a second
    instead of minutes: it is the only place ASR is triggered.
    """
    if not force:
        cached = load_index(media_key)
        if cached is not None:
            report(progress_callback, STAGE_CHECK,
                   f"cache hit: {cached.word_count} words, model={cached.model} "
                   f"(index v{cached.index_version})")
            return cached
        report(progress_callback, STAGE_CHECK, "no valid index on disk, running ASR")

    return build_index(media_key, wav_path, model_name=model_name,
                       language=language, progress_callback=progress_callback)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.core.index")
    parser.add_argument("media_key")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached")
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--text", action="store_true", help="Print the normalized text")
    args = parser.parse_args(argv)

    try:
        index = ensure_index(args.media_key, paths.audio_path(args.media_key),
                             force=args.force, model_name=args.model, language=args.language)
    except DialogueFrameError as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    print(f"\nmedia_key     : {index.media_key}")
    print(f"index_version : {index.index_version}")
    print(f"source        : {'cache' if index.from_cache else 'freshly built'}")
    print(f"model         : {index.model}")
    print(f"language      : {index.language} (p={index.language_probability or 0:.2f})")
    print(f"duration      : {index.duration:.2f}s")
    print(f"words         : {index.word_count} in {len(index.segments)} segments")
    print(f"normalized    : {len(index.normalized_text)} chars")
    print(f"asr elapsed   : {index.asr_elapsed_seconds:.1f}s")
    if args.text:
        print(f"\nnormalized_text:\n  {index.normalized_text}")
    print()
    for i, word in enumerate(index.words[:args.limit]):
        print(f"  {i:>4}  {word['start']:>8.2f}  {word['end']:>8.2f}  {word['word']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
