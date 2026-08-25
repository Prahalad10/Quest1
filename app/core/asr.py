"""Transcribe a wav to word-level timestamps with faster-whisper.

Knows nothing about videos, caching, or matching. Word timings are the entire
point: a frame number is only as precise as the word boundary it came from.

The expensive stage -- roughly 2-9x realtime on CPU depending on how densely
the audio is spoken, since VAD skips silence. That cost is why its output is
cached per video and never recomputed for a new query.

    python -m app.core.asr data/<media_key>/audio.wav --limit 20
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from app import config
from app.errors import InvalidInputError, Quest1Error
from app.progress import ProgressCallback, report

STAGE = "asr"


class AsrError(Quest1Error):
    """Model would not load, decode crashed, or the transcript had no words."""


def format_clock(seconds: float) -> str:
    """Seconds as M:SS, or H:MM:SS past an hour."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@dataclass
class Word:
    """One spoken word and the span it occupies."""

    word: str
    start: float
    end: float
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    """Whisper's sentence-level segmentation.

    Not used for matching -- that works on the flat word stream so a query can
    span a segment boundary. Kept because it is the readable form when
    debugging what the model actually heard.
    """

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


class _Heartbeat:
    """Emits interpolated progress between real segment events.

    faster-whisper only yields a segment once it has finished decoding it, and
    on long audio consecutive segments can be 30-40s apart. A bar driven purely
    by segment arrivals sits frozen in between and looks broken -- that was the
    reported "stuck at 80%".

    Two rules keep it honest. The estimate is clamped below the next real
    position, so it can lead slightly but never overshoot into a lie. And
    before the first segment there is no rate to extrapolate from, so it
    reports percent=None with a ticking clock rather than inventing a number.
    """

    def __init__(self, total: float, callback: Optional[ProgressCallback], interval: float):
        self._total = total
        self._callback = callback
        self._interval = max(0.5, interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = time.time()
        self._position = 0.0
        self._position_at = self._started_at
        self._words = 0
        self._seen_segment = False
        self._rate = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="quest1-asr-heartbeat", daemon=True)
        self._thread.start()

    def update(self, position: float, words: int) -> None:
        """Anchor to a position a real segment confirmed."""
        with self._lock:
            now = time.time()
            self._position = float(position)
            self._position_at = now
            self._words = words
            self._seen_segment = True
            span = now - self._started_at
            if span > 0:
                self._rate = float(position) / span

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)

    def _estimate(self) -> tuple[Optional[float], str]:
        with self._lock:
            position, position_at = self._position, self._position_at
            words, seen, rate = self._words, self._seen_segment, self._rate
        now = time.time()

        if not seen or not self._total or rate <= 0:
            return None, f"transcribing… {format_clock(now - self._started_at)} elapsed"

        # The rate is NOT recomputed here. An earlier version used
        # position/elapsed every tick; between segments position is frozen
        # while elapsed grows, so the rate collapsed and the ETA climbed from
        # 16 to 39 minutes while the position barely moved.
        projected = min(position + (now - position_at) * rate, self._total)
        remaining = (self._total - projected) / rate
        eta = f", ~{format_clock(remaining)} left" if remaining > 2 else ""
        return min(100.0, projected / self._total * 100), (
            f"{format_clock(projected)} / {format_clock(self._total)} transcribed, "
            f"{words} words{eta}"
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            percent, message = self._estimate()
            report(self._callback, STAGE, message, percent=percent)


def _load_model(model_name: str, device: str, compute_type: str):
    """Imported lazily -- faster_whisper pulls in ctranslate2 and onnxruntime,
    which costs a second or two that the not-found path should not pay."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - environment problem
        raise AsrError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001 - download, RAM, bad name, bad compute type
        raise AsrError(
            f"Could not load Whisper model {model_name!r} on {device}/{compute_type}: "
            f"{type(exc).__name__}: {exc}. The first run downloads the model, so check your "
            f"connection; set QUEST1_ASR_MODEL to a smaller model if RAM is the issue."
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
    """Transcribe with word timestamps and VAD.

    word_timestamps=True is non-negotiable: without it Whisper returns only
    sentence spans, far too coarse to pick a frame from. vad_filter trims
    silence, which both speeds up decoding and stops hallucination over long
    quiet stretches.
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

    report(progress_callback, STAGE,
           f"loading model {model_name} ({device}/{compute_type}) -- first run downloads it")
    model = _load_model(model_name, device, compute_type)

    report(progress_callback, STAGE, f"transcribing {wav.name} (vad={vad_filter}, beam={beam_size})")
    started = time.time()
    try:
        segment_iter, info = model.transcribe(
            str(wav), word_timestamps=True, vad_filter=vad_filter,
            beam_size=beam_size, language=language,
        )
    except Exception as exc:  # noqa: BLE001
        raise AsrError(f"Transcription failed: {type(exc).__name__}: {exc}") from exc

    total = float(getattr(info, "duration", 0.0) or 0.0)
    segments: list[Segment] = []
    words: list[Word] = []
    heartbeat = _Heartbeat(total, progress_callback, config.ASR_PROGRESS_INTERVAL_SECONDS)
    heartbeat.start()

    try:
        # faster-whisper decodes lazily: the real work happens here.
        for index, segment in enumerate(segment_iter):
            segments.append(Segment(id=index, start=float(segment.start),
                                    end=float(segment.end), text=(segment.text or "").strip()))
            for word in (segment.words or []):
                # A null timing cannot locate a frame, so drop the word rather
                # than invent a timestamp for it.
                if word.start is None or word.end is None:
                    continue
                text = (word.word or "").strip()
                if not text:
                    continue
                words.append(Word(word=text, start=float(word.start), end=float(word.end),
                                  probability=float(getattr(word, "probability", 0.0) or 0.0)))
            heartbeat.update(float(segment.end), len(words))
    except Exception as exc:  # noqa: BLE001
        raise AsrError(f"Transcription failed mid-stream: {type(exc).__name__}: {exc}") from exc
    finally:
        # Must stop on every path, or it keeps reporting after the stage moved
        # on and the bar jumps backwards.
        heartbeat.stop()

    elapsed = time.time() - started
    if not words:
        raise AsrError(
            f"No words were transcribed from {wav}. The audio may be silent, music-only, or in "
            f"a language the model could not decode (detected: {getattr(info, 'language', '?')})."
        )

    report(progress_callback, STAGE,
           f"done: {len(words)} words in {len(segments)} segments, "
           f"language={getattr(info, 'language', '?')}, {elapsed:.1f}s")
    return Transcription(
        segments=segments, words=words,
        language=getattr(info, "language", None),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=total, model=model_name, elapsed_seconds=elapsed,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Shows what the model actually heard -- the first place to look when a
    phrase is not found, before touching MATCH_THRESHOLD."""
    parser = argparse.ArgumentParser(prog="python -m app.core.asr")
    parser.add_argument("wav", help="Path to a 16kHz mono wav (see app.core.audio)")
    parser.add_argument("--limit", type=int, default=25, help="How many words to print")
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default=None)
    args = parser.parse_args(argv)

    try:
        result = transcribe(args.wav, model_name=args.model, language=args.language)
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    print(f"\nmodel     : {result.model}")
    print(f"language  : {result.language} (p={result.language_probability:.2f})")
    print(f"duration  : {result.duration:.2f}s")
    print(f"words     : {len(result.words)} in {len(result.segments)} segments")
    print(f"elapsed   : {result.elapsed_seconds:.1f}s "
          f"({result.duration / max(result.elapsed_seconds, 1e-9):.2f}x realtime)\n")
    for i, word in enumerate(result.words[:args.limit]):
        print(f"  {i:>4}  {word.start:>8.2f}  {word.end:>8.2f}  {word.probability:>5.2f}  {word.word}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
