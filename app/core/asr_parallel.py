"""Transcribe one wav across several processes at once.

WHY THIS MODULE EXISTS
    app/core/asr.py decodes a wav in ONE process, and one process cannot use
    this machine. Measured on 90s of dense speech (small, int8, beam 5), varying
    only the threads given to that single stream:

        1 thread  111.1s   0.81x realtime
        2 threads  82.2s   1.10x realtime
        4 threads  69.4s   1.30x realtime
        8 threads  76.4s   1.18x realtime   (slower -- 4 physical cores only)

    Quadrupling the threads buys 1.6x. Whisper's decoder is autoregressive -- it
    predicts one token at a time -- so the extra cores have little to do on a
    single stream. Giving each core a DIFFERENT piece of audio instead yields
    4 x 0.81 = 3.24x realtime aggregate, a 2.5x speedup over the same hardware
    running one stream. That is what this module does.

    It is a large constant-factor win, not a change of order. See section 11 of
    Constants.txt for what feature-length audio still costs afterwards.

WHAT IT GUARANTEES
    The returned Transcription is the same type app/core/asr.py returns, with the
    same absolute (whole-file) timestamps, so app/core/index.py cannot tell which
    path produced it. That is deliberate: the cache format, the matcher and the
    frame lookup stay completely unaware of parallelism.

THE SEAM PROBLEM
    Cutting audio at an arbitrary second can cut a word in half, and a decoder
    started mid-sentence has no context. Both are handled by decoding OVERLAPPING
    chunks and then attributing every word to exactly one chunk by its start
    time -- the overlap is context for the decoder, never extra output. See
    _merge_chunks.

Run directly:
    python -m app.core.asr_parallel data/<media_key>/audio.wav --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Optional

from app import config
from app.core.asr import AsrError, Segment, Transcription, Word, format_clock
from app.errors import InvalidInputError, Quest1Error
from app.progress import ProgressCallback, report

STAGE = "asr"


# --------------------------------------------------------------------------- #
# Audio slicing
# --------------------------------------------------------------------------- #

def wav_duration_seconds(wav_path: Path | str) -> float:
    """Exact duration of a PCM wav, from its header rather than by decoding it.

    WHY NOT ffprobe: this is called on every parallel run just to decide how many
    chunks to cut, and spawning a process to learn a number already in the file
    header is wasteful. WHY NOT the probe cache: that describes the VIDEO, whose
    duration can differ slightly from the extracted wav's.

    USED BY: transcribe_parallel, to plan chunk boundaries.
    """
    with wave.open(str(wav_path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    if rate <= 0:
        raise AsrError(f"{wav_path} reports a sample rate of {rate}")
    return frames / float(rate)


def _read_slice(wav_path: str, start: float, duration: float):
    """Read [start, start+duration) from a wav as float32 mono, without the rest.

    WHY A SLICE READER: a 90-minute wav at 16kHz mono is ~170 MB. Handing every
    worker the whole array would copy that per process. wave.setpos seeks
    directly to the sample, so each worker pays only for its own chunk.

    Returns the numpy array faster-whisper accepts in place of a path.

    USED BY: _transcribe_chunk, inside the worker process.
    """
    import numpy as np

    with wave.open(wav_path, "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        if channels != 1 or width != 2:
            raise AsrError(
                f"Parallel ASR needs 16-bit mono audio, got {channels}ch/{width * 8}-bit. "
                f"app.core.audio should have produced that; the wav may be stale."
            )
        first = max(0, int(start * rate))
        count = max(0, int(duration * rate))
        handle.setpos(min(first, handle.getnframes()))
        raw = handle.readframes(count)

    # int16 -> float32 in [-1, 1) is exactly what faster-whisper does internally
    # when handed a path, so doing it here changes nothing about the decode.
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def _transcribe_chunk(task: dict[str, Any]) -> dict[str, Any]:
    """Decode one chunk. RUNS IN A SEPARATE PROCESS -- keep it picklable.

    Every argument arrives as a plain dict and every result leaves as one,
    because ProcessPoolExecutor pickles both and this project's dataclasses are
    not worth the round trip. Timestamps are shifted to whole-file time HERE, so
    the parent never has to remember which chunk a word came from.

    WHY IT LOADS ITS OWN MODEL: a ctranslate2 model cannot be shared across
    processes. The load is a few seconds and happens concurrently in every
    worker, so it costs wall time once, not once per chunk.

    USED BY: transcribe_parallel, via ProcessPoolExecutor.
    """
    from faster_whisper import WhisperModel

    offset = float(task["start"])
    audio = _read_slice(task["wav"], offset, float(task["duration"]))
    if audio.size == 0:
        return {"index": task["index"], "segments": [], "words": [],
                "language": None, "language_probability": 0.0, "audio_seconds": 0.0}

    model = WhisperModel(
        task["model_name"],
        device=task["device"],
        compute_type=task["compute_type"],
        cpu_threads=int(task["cpu_threads"]),
    )
    segment_iter, info = model.transcribe(
        audio,
        word_timestamps=True,        # the whole reason this project works
        vad_filter=bool(task["vad_filter"]),
        beam_size=int(task["beam_size"]),
        language=task["language"],
    )

    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for segment in segment_iter:
        segments.append({
            "start": float(segment.start) + offset,
            "end": float(segment.end) + offset,
            "text": (segment.text or "").strip(),
        })
        for word in (segment.words or []):
            # A word with a null timing cannot locate a frame, so it is dropped
            # rather than given an invented timestamp.
            if word.start is None or word.end is None:
                continue
            text = (word.word or "").strip()
            if not text:
                continue
            words.append({
                "word": text,
                "start": float(word.start) + offset,
                "end": float(word.end) + offset,
                "probability": float(getattr(word, "probability", 0.0) or 0.0),
            })

    return {
        "index": task["index"],
        "segments": segments,
        "words": words,
        "language": getattr(info, "language", None),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "audio_seconds": float(audio.size) / config.AUDIO_SAMPLE_RATE,
    }


# --------------------------------------------------------------------------- #
# Planning and merging
# --------------------------------------------------------------------------- #

def plan_chunks(
    duration: float, chunk_seconds: float, overlap_seconds: float
) -> list[tuple[float, float, float, float]]:
    """Cut [0, duration) into decode windows plus the region each one OWNS.

    Returns (decode_start, decode_duration, own_start, own_end) per chunk.

    The decode window is longer than the owned region by `overlap_seconds`: the
    extra audio is context so the decoder does not start cold mid-sentence and
    does not truncate a word sitting on the boundary. Output from that extra
    audio is discarded -- ownership, not decode extent, decides which chunk a
    word comes from, which is what makes the merge free of duplicates.

    USED BY: transcribe_parallel and its self-test.
    """
    if duration <= 0:
        return []
    chunk_seconds = max(10.0, float(chunk_seconds))
    overlap_seconds = max(0.0, float(overlap_seconds))
    count = max(1, math.ceil(duration / chunk_seconds))
    plan: list[tuple[float, float, float, float]] = []
    for i in range(count):
        own_start = i * chunk_seconds
        own_end = min(duration, own_start + chunk_seconds)
        if own_start >= duration:
            break
        # Lead-in gives context for the first word of the region; lead-out keeps
        # a word that straddles the far boundary from being cut in half.
        decode_start = max(0.0, own_start - overlap_seconds)
        decode_end = min(duration, own_end + overlap_seconds)
        plan.append((decode_start, decode_end - decode_start, own_start, own_end))
    return plan


def _word_key(text: str) -> str:
    """Letters and digits of a word, lowercased.

    WHY: so "Watson," and "Watson" compare equal when checking whether two
    chunks reported the same utterance. USED BY: _merge_chunks.
    """
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _merge_chunks(
    results: list[dict[str, Any]], plan: list[tuple[float, float, float, float]]
) -> tuple[list[Segment], list[Word]]:
    """Stitch per-chunk output into one whole-file transcript.

    TWO DEFENCES AGAINST DOUBLE-COUNTING A SEAM:
      1. Ownership. A word is kept only by the chunk whose owned region contains
         its start time. Since the regions tile [0, duration) without gaps or
         overlap, every word belongs to exactly one chunk.
      2. A temporal duplicate check. Ownership alone can still let the same
         spoken word through twice if the two decoders placed it a few
         milliseconds either side of the boundary, so a word that starts before
         the previous kept word ended AND reads the same is dropped.

    WHY IT MATTERS: a duplicated word would shift every character offset after
    it in the flat text app/core/index.py builds, and the matcher would then
    report a timestamp for the wrong word.

    USED BY: transcribe_parallel.
    """
    words: list[Word] = []
    segments: list[Segment] = []

    for result in sorted(results, key=lambda r: r["index"]):
        own_start, own_end = plan[result["index"]][2], plan[result["index"]][3]
        # The last chunk owns everything to the end, so a word the decoder
        # placed a hair past the nominal end is not silently lost.
        is_last = result["index"] == len(plan) - 1
        for word in result["words"]:
            if word["start"] < own_start:
                continue
            if not is_last and word["start"] >= own_end:
                continue
            if (words and word["start"] < words[-1].end
                    and _word_key(word["word"]) == _word_key(words[-1].word)):
                continue
            words.append(Word(
                word=word["word"], start=word["start"],
                end=word["end"], probability=word["probability"],
            ))
        for segment in result["segments"]:
            if segment["start"] < own_start or (not is_last and segment["start"] >= own_end):
                continue
            segments.append(Segment(
                id=len(segments), start=segment["start"],
                end=segment["end"], text=segment["text"],
            ))

    return segments, words


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def worker_count(duration: float) -> int:
    """How many processes to use for a `duration`-second wav.

    Capped by the configured maximum, by the core count, and by how many chunks
    the audio even has -- starting eight workers for three chunks would pay
    eight model loads to do three chunks' worth of work.

    USED BY: transcribe_parallel, and should_parallelize.
    """
    chunks = max(1, math.ceil(duration / config.ASR_CHUNK_SECONDS))
    cores = os.cpu_count() or 2
    return max(1, min(config.ASR_MAX_WORKERS, cores, chunks))


def should_parallelize(duration: float) -> bool:
    """Whether parallel decoding is worth its overhead for this audio.

    Each worker pays a model load of a few seconds. Below a threshold the serial
    path in app/core/asr.py finishes sooner than the loads alone would cost, so
    short clips deliberately stay serial.

    USED BY: app/core/index.py, to pick a transcription path.
    """
    return (
        config.ASR_PARALLEL
        and duration >= config.ASR_PARALLEL_MIN_SECONDS
        and worker_count(duration) > 1
    )


def _prefetch_model(model_name: str, progress_callback: Optional[ProgressCallback]) -> None:
    """Make sure the model files are on disk before the workers start.

    WHY: on a first run every worker would otherwise hit the Hugging Face cache
    simultaneously for the same files. Downloading once here turns that into N
    cache hits. Failures are non-fatal -- the workers would download it anyway,
    and their error message is the more useful one.

    USED BY: transcribe_parallel.
    """
    try:
        from faster_whisper.utils import download_model
        report(progress_callback, STAGE, f"ensuring model {model_name} is available")
        download_model(model_name)
    except Exception:  # noqa: BLE001 - best effort only
        pass



def detect_language(
    wav_path: Path | str,
    duration: float,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> Optional[str]:
    """Detect the spoken language ONCE, in the parent, from a sample of the audio.

    WHY THIS IS NOT LEFT TO THE WORKERS
        With language=None every chunk detects independently. On a feature-length
        film that is thirty-odd separate guesses, and they do not have to agree.
        A chunk that happens to be music, wind or silence can be assigned some
        unrelated language, and Whisper will then hallucinate fluent text in it
        -- which lands in the transcript as real dialogue at a real timestamp.
        Deciding once and telling every worker removes that failure mode
        entirely, and skips ~N-1 redundant detections.

    WHY THE SAMPLE IS NOT AT THE START: opening titles and logos are usually
    music over no speech, which is the worst possible sample. A window a third
    of the way in is almost always dialogue.

    Returns None if detection fails, which leaves the workers to auto-detect --
    the previous behaviour, so a failure here degrades rather than breaks.

    USED BY: transcribe_parallel, when no language was configured.
    """
    try:
        from faster_whisper import WhisperModel

        sample_start = max(0.0, min(duration * 0.33, max(0.0, duration - 30.0)))
        audio = _read_slice(str(wav_path), sample_start, 30.0)
        if audio.size == 0:
            return None
        model = WhisperModel(model_name, device=device, compute_type=compute_type,
                             cpu_threads=max(1, config.ASR_WORKER_THREADS))
        # transcribe() reports what it detected on `info`; asking for one segment
        # is the cheapest way to get that without a separate API.
        _segments, info = model.transcribe(audio, beam_size=1, word_timestamps=False,
                                           vad_filter=False)
        language = getattr(info, "language", None)
        probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        if language:
            report(progress_callback, STAGE,
                   f"language detected once for all chunks: {language} (p={probability:.2f})")
        return language
    except Exception:  # noqa: BLE001 - detection is an optimisation, not a requirement
        return None


def transcribe_parallel(
    wav_path: Path | str,
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    beam_size: Optional[int] = None,
    vad_filter: Optional[bool] = None,
    workers: Optional[int] = None,
    chunk_seconds: Optional[float] = None,
    overlap_seconds: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Transcription:
    """Transcribe a wav across processes and return one whole-file Transcription.

    Drop-in replacement for app.core.asr.transcribe with an identical output type
    and identical absolute timestamps. Raises AsrError if any chunk fails or if
    the merged transcript has no words.

    USED BY: app/core/index.py:build_index, when should_parallelize() says so.
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
    chunk_seconds = config.ASR_CHUNK_SECONDS if chunk_seconds is None else chunk_seconds
    overlap_seconds = (
        config.ASR_CHUNK_OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds
    )

    duration = wav_duration_seconds(wav)
    plan = plan_chunks(duration, chunk_seconds, overlap_seconds)
    if not plan:
        raise AsrError(f"{wav} contains no audio to transcribe.")
    n_workers = workers or worker_count(duration)
    # One thread per worker by default. Measured: a single stream gains almost
    # nothing from extra threads, so the cores are better spent on more workers.
    # See the ASR_WORKER_THREADS note in app/config.py for the numbers.
    cpu_threads = max(1, config.ASR_WORKER_THREADS)

    report(
        progress_callback, STAGE,
        f"transcribing {format_clock(duration)} of audio in {len(plan)} chunks "
        f"across {n_workers} workers (model={model_name}, beam={beam_size})",
        percent=0.0,
    )
    _prefetch_model(model_name, progress_callback)
    if language is None:
        language = detect_language(
            wav, duration, model_name=model_name,
            device=device, compute_type=compute_type,
            progress_callback=progress_callback,
        )

    tasks = [
        {
            "index": i, "wav": str(wav), "start": start, "duration": length,
            "model_name": model_name, "device": device, "compute_type": compute_type,
            "beam_size": beam_size, "vad_filter": vad_filter, "language": language,
            "cpu_threads": cpu_threads,
        }
        for i, (start, length, _own_start, _own_end) in enumerate(plan)
    ]

    started = time.time()
    results: list[dict[str, Any]] = []
    interval = max(0.5, config.ASR_PROGRESS_INTERVAL_SECONDS)
    owned_total = sum(own_end - own_start for _s, _d, own_start, own_end in plan)
    owned_done = 0.0

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            pending = {pool.submit(_transcribe_chunk, task) for task in tasks}
            while pending:
                # A timeout rather than a plain as_completed loop: it lets the
                # progress bar tick between chunk completions, which on a long
                # video can be a minute apart. A bar that stops moving for a
                # minute reads as broken -- that was a real reported bug.
                finished, pending = concurrent.futures.wait(
                    pending, timeout=interval,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in finished:
                    result = future.result()   # re-raises a worker exception here
                    results.append(result)
                    idx = result["index"]
                    owned_done += plan[idx][3] - plan[idx][2]
                elapsed = time.time() - started
                percent = min(100.0, owned_done / owned_total * 100) if owned_total else None
                eta = ""
                if owned_done > 0:
                    remaining = (owned_total - owned_done) * (elapsed / owned_done)
                    if remaining > 2:
                        eta = f", ~{format_clock(remaining)} left"
                report(
                    progress_callback, STAGE,
                    f"{len(results)}/{len(tasks)} chunks done "
                    f"({format_clock(owned_done)} / {format_clock(duration)}){eta}",
                    percent=percent,
                )
    except Exception as exc:  # noqa: BLE001 - a worker can fail many ways
        if isinstance(exc, Quest1Error):
            raise
        raise AsrError(
            f"Parallel transcription failed: {type(exc).__name__}: {exc}. "
            f"Set QUEST1_ASR_PARALLEL=0 to fall back to single-process ASR."
        ) from exc

    segments, words = _merge_chunks(results, plan)
    elapsed = time.time() - started
    if not words:
        raise AsrError(
            f"No words were transcribed from {wav}. The audio may be silent, "
            f"music-only, or in a language the model could not decode."
        )

    # Language is whatever the majority of chunks detected; chunks of the same
    # film essentially always agree, and a single odd chunk should not rename
    # the whole transcript.
    votes: dict[str, int] = {}
    for result in results:
        if result["language"]:
            votes[result["language"]] = votes.get(result["language"], 0) + 1
    language_detected = max(votes, key=lambda k: votes[k]) if votes else None
    probs = [r["language_probability"] for r in results if r["language"] == language_detected]

    report(
        progress_callback, STAGE,
        f"done: {len(words)} words in {len(segments)} segments across {len(plan)} chunks, "
        f"language={language_detected}, {elapsed:.1f}s "
        f"({duration / max(elapsed, 1e-9):.2f}x realtime)",
        percent=100.0,
    )
    return Transcription(
        segments=segments,
        words=words,
        language=language_detected,
        language_probability=(sum(probs) / len(probs)) if probs else 0.0,
        duration=duration,
        model=model_name,
        elapsed_seconds=elapsed,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry point: time a parallel transcription of one wav.

    WHY: lets the parallel path be measured and compared against
    `python -m app.core.asr` on the same file without running the pipeline.

    USED BY: `python -m app.core.asr_parallel <wav> --workers 4`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.core.asr_parallel",
        description="Transcribe a wav across several processes.",
    )
    parser.add_argument("wav", help="Path to a 16kHz mono wav (see app.core.audio)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk", type=float, default=None, help="Chunk seconds")
    parser.add_argument("--model", default=None)
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args(argv)

    from app.progress import stderr_progress
    try:
        result = transcribe_parallel(
            args.wav, workers=args.workers, chunk_seconds=args.chunk,
            model_name=args.model, beam_size=args.beam,
            progress_callback=stderr_progress,
        )
    except Quest1Error as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"model     : {result.model}")
    print(f"language  : {result.language} (p={result.language_probability:.2f})")
    print(f"duration  : {result.duration:.2f}s")
    print(f"segments  : {len(result.segments)}")
    print(f"words     : {len(result.words)}")
    print(f"elapsed   : {result.elapsed_seconds:.1f}s "
          f"({result.duration / max(result.elapsed_seconds, 1e-9):.2f}x realtime)")
    print()
    for i, word in enumerate(result.words[:args.limit]):
        print(f"  {i:>4}  {word.start:>8.2f}  {word.end:>8.2f}  {word.word}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
