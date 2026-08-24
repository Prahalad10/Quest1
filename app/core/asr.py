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
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from app import config
from app.errors import InvalidInputError, Quest1Error
from app.progress import ProgressCallback, report

STAGE = "asr"


def format_clock(seconds: float) -> str:
    """Format a number of seconds as M:SS, or H:MM:SS past an hour.

    WHY: progress messages read far better as "3:42 / 8:50" than as
    "222.4s / 530.1s", especially on a long video.

    USED BY: the ASR progress messages below.
    """
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class _Heartbeat:
    """Emits interpolated ASR progress between real segment events.

    WHY THIS EXISTS
        faster-whisper's generator only yields once a segment has finished
        decoding. On a long video consecutive segments can be 30-40 seconds
        apart, so a progress bar driven purely by segment arrivals sits
        completely frozen in between and looks broken -- which is exactly what
        a 10-minute video showed: the bar parked at ~80% for minutes.

        This background thread fills those gaps. Between segments it estimates
        the current decode position from the observed rate so far, and reports
        that instead. The estimate is always clamped BELOW the next real
        position, so it can lead slightly but never overshoot into a lie.

    HONESTY RULE
        Before the first segment arrives there is no rate to extrapolate from,
        so it reports percent=None (indeterminate) with the elapsed time in the
        message. An indeterminate bar plus a ticking clock reads as "working";
        a fabricated percentage would read as "stuck" the moment it stalled.

    USED BY: transcribe().
    """

    def __init__(
        self,
        total: float,
        callback: Optional[ProgressCallback],
        interval: float,
    ) -> None:
        """Prepare (but do not start) the heartbeat for a `total`-second audio."""
        self._total = total
        self._callback = callback
        self._interval = max(0.5, interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = time.time()
        # Anchor: the last position a real segment confirmed, and when.
        self._position = 0.0
        self._position_at = self._started_at
        self._words = 0
        self._seen_segment = False
        # Decode rate in audio-seconds per wall-second, recomputed ONLY when a
        # real segment lands. See _estimate for why it must not be recomputed
        # on every tick.
        self._rate = 0.0

    def start(self) -> None:
        """Begin emitting. Daemon so it can never hold up interpreter exit."""
        self._thread = threading.Thread(
            target=self._run, name="quest1-asr-heartbeat", daemon=True
        )
        self._thread.start()

    def update(self, position: float, words: int) -> None:
        """Record a real segment boundary. CALLED FROM: the transcription loop."""
        with self._lock:
            now = time.time()
            self._position = float(position)
            self._position_at = now
            self._words = words
            self._seen_segment = True
            # Rate is measured against the time this position was CONFIRMED,
            # so it reflects actual throughput and stays stable between
            # segments instead of decaying while we wait for the next one.
            span = now - self._started_at
            if span > 0:
                self._rate = float(position) / span

    def stop(self) -> None:
        """Stop emitting and wait briefly for the thread to notice."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)

    def _estimate(self) -> tuple[Optional[float], str]:
        """Return (percent, message) for right now."""
        with self._lock:
            position, position_at = self._position, self._position_at
            words, seen, rate = self._words, self._seen_segment, self._rate
        now = time.time()
        elapsed = now - self._started_at

        if not seen or not self._total or rate <= 0:
            # No confirmed rate yet: indeterminate, but visibly alive.
            return None, f"transcribing… {format_clock(elapsed)} elapsed"

        # WHY THE RATE IS NOT RECOMPUTED HERE: an earlier version used
        # position/elapsed on every tick. Between segments `position` is frozen
        # while `elapsed` keeps growing, so the rate collapsed and the ETA grew
        # the longer you waited -- it climbed from 16 to 39 minutes while the
        # position barely moved, which reads as broken. The rate confirmed by
        # the last real segment is held steady instead.
        projected = min(position + (now - position_at) * rate, self._total)

        percent = min(100.0, projected / self._total * 100)
        remaining = (self._total - projected) / rate
        eta = f", ~{format_clock(remaining)} left" if remaining > 2 else ""
        return percent, (
            f"{format_clock(projected)} / {format_clock(self._total)} transcribed, "
            f"{words} words{eta}"
        )

    def _run(self) -> None:
        """Thread body: emit an estimate every `interval` until stopped."""
        while not self._stop.wait(self._interval):
            percent, message = self._estimate()
            report(self._callback, STAGE, message, percent=percent)


class AsrError(Quest1Error):
    """Transcription failed or produced unusable output.

    Covers a model that will not load, a decode that crashes mid-stream, and a
    transcript that came back with zero words (silent or music-only audio).

    RAISED BY: this module. CAUGHT BY: app/cli.py and app/api.py as a Quest1Error.
    """


@dataclass
class Word:
    """One spoken word with the time span it occupies.

    The start/end times are the entire reason this project works: a frame number
    is only ever as precise as the word boundary it was derived from.

    USED BY: index.build_flat_text, which normalizes each word and records its
    character offsets, then persists these to transcript.json.
    """

    word: str          # as Whisper emitted it, minus its leading space
    start: float
    end: float
    probability: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form. USED BY: index.py when writing transcript.json."""
        return asdict(self)


@dataclass
class Segment:
    """Whisper's own sentence-level segmentation.

    Not used for matching -- matching works on the flat word stream so a query
    can span a segment boundary. Kept because it is the readable form of the
    transcript when debugging why a phrase did or did not match.

    USED BY: index.py, which persists these to transcript.json unchanged.
    """

    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, used wherever this is persisted or returned."""
        return asdict(self)


@dataclass
class Transcription:
    """Everything one ASR run produced.

    Returned by transcribe() and immediately converted into a TranscriptIndex by
    index.build_index; nothing else consumes this type directly.
    """

    segments: list[Segment]
    words: list[Word]
    language: Optional[str]
    language_probability: Optional[float]
    duration: float
    model: str
    elapsed_seconds: float


def _load_model(model_name: str, device: str, compute_type: str):
    """Import and construct the model, turning setup failures into clear errors.

    WHY THE IMPORT IS INSIDE A FUNCTION: importing faster_whisper pulls in
    ctranslate2 and onnxruntime, which costs a second or two. Doing it lazily
    keeps `python -m app.core.resolve` and the not-found query path fast, since
    neither needs a model.

    WHY THE BROAD EXCEPT: model construction can fail on a download error, a
    missing model name, insufficient RAM, or an unsupported compute type. All of
    them need the same actionable message rather than a raw traceback.

    USED BY: transcribe(), below.
    """
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

    The single expensive operation in the whole pipeline: on CPU this runs at
    roughly 0.7x realtime, so a 10-minute video costs about 7 minutes. That cost
    is exactly why its output is cached per video and never recomputed for a new
    query -- see index.ensure_index.

    word_timestamps=True is non-negotiable: without it Whisper returns only
    sentence-level spans, which are far too coarse to pick a frame from.
    vad_filter=True trims silence, which both speeds up decoding and stops
    Whisper hallucinating text over long quiet stretches.

    Raises InvalidInputError if the wav is missing, AsrError if the model cannot
    load or the transcript comes back with no words.

    USED BY: index.build_index. Also runnable standalone via the __main__ block
    for checking what the model actually heard.
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

    # Keeps the progress bar moving between segment arrivals; see _Heartbeat.
    heartbeat = _Heartbeat(total, progress_callback, config.ASR_PROGRESS_INTERVAL_SECONDS)
    heartbeat.start()

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
            # Anchor the heartbeat to a position we actually reached.
            heartbeat.update(float(segment.end), len(words))
    except Exception as exc:  # noqa: BLE001
        raise AsrError(f"Transcription failed mid-stream: {type(exc).__name__}: {exc}") from exc
    finally:
        # Must stop in every path, or the thread keeps reporting after the stage
        # has moved on and the bar jumps backwards.
        heartbeat.stop()

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
    """Standalone entry point: transcribe a wav and print the first N words.

    WHY: lets you check ASR quality on its own, without running resolve, audio
    fetch, matching or frame extraction. If a phrase is not being found, this is
    the first place to look -- it shows exactly what the model heard.

    USED BY: `python -m app.core.asr <wav> --limit 20`.
    """
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
