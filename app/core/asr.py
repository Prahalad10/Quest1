"""Transcribe audio to word-level timestamps with faster-whisper.

This module knows nothing about videos, caching, or matching. It takes a wav
path and returns segments plus a flat word list with per-word start/end times --
the timings are the entire point, since a frame number is only as precise as the
word boundary it came from.

Run directly:
    python -m app.core.asr data/<media_key>/audio.wav --limit 20
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from app import config
from app.errors import InvalidInputError, Quest1Error
from app.progress import ProgressCallback, report

STAGE = "asr"


class AsrError(Quest1Error):
    """Transcription failed or produced unusable output."""


@dataclass
class Word:
    """One spoken word with the time span it occupies."""

    word: str          # as Whisper emitted it, minus its leading space
    start: float
    end: float
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    """Whisper's own segmentation, kept for context display and debugging."""

    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transcription:
    segments: list[Segment]
    words: list[Word]
    language: Optional[str]
    language_probability: Optional[float]
    duration: float
    model: str
    elapsed_seconds: float


def _load_model(model_name: str, device: str, compute_type: str):
    """Import and construct the model, turning setup failures into clear errors."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - environment problem
        raise AsrError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001 - model load has many failure modes
        raise AsrError(
            f"Could not load Whisper model {model_name!r} on {device}/{compute_type}: "
            f"{type(exc).__name__}: {exc}. The first run downloads the model, so check "
            f"your connection; set QUEST1_ASR_MODEL to a smaller model if RAM is the issue."
        ) from exc


def transcribe(
    wav_path: Path | str,
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    beam_size: Optional[int] = None,
    vad_filter: Optional[bool] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Transcription:
    """Transcribe `wav_path` with word timestamps and VAD.

    Raises InvalidInputError if the wav is missing, AsrError if the model cannot
    load or the transcript comes back with no words.
    """
    wav = Path(wav_path)
    if not wav.exists():
        raise InvalidInputError(f"Audio file not found: {wav}. Run app.core.audio first.")
    if wav.stat().st_size == 0:
        raise InvalidInputError(f"Audio file is empty: {wav}")

    model_name = model_name or config.ASR_MODEL
    device = device or config.ASR_DEVICE
    compute_type = compute_type or config.ASR_COMPUTE_TYPE
    beam_size = config.ASR_BEAM_SIZE if beam_size is None else beam_size
    vad_filter = config.ASR_VAD_FILTER if vad_filter is None else vad_filter
    language = language if language is not None else config.ASR_LANGUAGE

    report(
        progress_callback, STAGE,
        f"loading model {model_name} ({device}/{compute_type}) -- first run downloads it",
    )
    model = _load_model(model_name, device, compute_type)

    report(
        progress_callback, STAGE,
        f"transcribing {wav.name} (vad={vad_filter}, beam={beam_size})",
    )
    started = time.time()
    try:
        segment_iter, info = model.transcribe(
            str(wav),
            word_timestamps=True,      # the whole reason this project works
            vad_filter=vad_filter,
            beam_size=beam_size,
            language=language,
        )
    except Exception as exc:  # noqa: BLE001
        raise AsrError(f"Transcription failed: {type(exc).__name__}: {exc}") from exc

    total = float(getattr(info, "duration", 0.0) or 0.0)
    segments: list[Segment] = []
    words: list[Word] = []
    last_reported = 0.0

    # faster-whisper decodes lazily: the real work happens as this is consumed.
    try:
        for index, segment in enumerate(segment_iter):
            segments.append(Segment(
                id=index,
                start=float(segment.start),
                end=float(segment.end),
                text=(segment.text or "").strip(),
            ))
            for word in (segment.words or []):
                # Whisper can emit a word with a null timing; it is unusable for
                # frame lookup, so drop it rather than invent a timestamp.
                if word.start is None or word.end is None:
                    continue
                text = (word.word or "").strip()
                if not text:
                    continue
                words.append(Word(
                    word=text,
                    start=float(word.start),
                    end=float(word.end),
                    probability=float(getattr(word, "probability", 0.0) or 0.0),
                ))
            if total and segment.end - last_reported >= max(5.0, total / 20):
                last_reported = float(segment.end)
                percent = min(100.0, segment.end / total * 100)
                report(
                    progress_callback, STAGE,
                    f"{percent:5.1f}%  ({segment.end:.1f}s / {total:.1f}s, {len(words)} words)",
                )
    except Exception as exc:  # noqa: BLE001
        raise AsrError(f"Transcription failed mid-stream: {type(exc).__name__}: {exc}") from exc

    elapsed = time.time() - started
    if not words:
        detected = getattr(info, "language", "unknown")
        raise AsrError(
            f"No words were transcribed from {wav}. The audio may be silent, music-only, "
            f"or in a language the model could not decode (detected: {detected})."
        )

    report(
        progress_callback, STAGE,
        f"done: {len(words)} words in {len(segments)} segments, "
        f"language={getattr(info, 'language', '?')}, {elapsed:.1f}s",
    )
    return Transcription(
        segments=segments,
        words=words,
        language=getattr(info, "language", None),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=total,
        model=model_name,
        elapsed_seconds=elapsed,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.core.asr",
        description="Transcribe a wav file to word-level timestamps.",
    )
    parser.add_argument("wav", help="Path to a 16kHz mono wav (see app.core.audio)")
    parser.add_argument("--limit", type=int, default=25, help="How many words to print")
    parser.add_argument("--model", default=None, help=f"Model name (default {config.ASR_MODEL})")
    parser.add_argument("--language", default=None, help="Force a language code, e.g. en")
    args = parser.parse_args(argv)

    try:
        result = transcribe(args.wav, model_name=args.model, language=args.language)
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"model     : {result.model}")
    print(f"language  : {result.language} (p={result.language_probability:.2f})")
    print(f"duration  : {result.duration:.2f}s")
    print(f"segments  : {len(result.segments)}")
    print(f"words     : {len(result.words)}")
    print(f"elapsed   : {result.elapsed_seconds:.1f}s")
    print()
    shown = min(args.limit, len(result.words))
    print(f"first {shown} words:")
    print(f"  {'#':>4}  {'START':>8}  {'END':>8}  {'PROB':>5}  WORD")
    for i, word in enumerate(result.words[:args.limit]):
        print(
            f"  {i:>4}  {word.start:>8.2f}  {word.end:>8.2f}  "
            f"{word.probability:>5.2f}  {word.word}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
