#!/usr/bin/env python
"""Multi-video accuracy and progress test harness.

WHY THIS EXISTS
    The pipeline was validated on a single 19-second video, which is not enough
    to trust either the matching thresholds or the progress reporting. A 19s clip
    has one ASR segment and almost no silence; a 10-minute one has dozens of
    segments, music beds, and speakers who trail off. This runs a spread of real
    videos and reports numbers rather than impressions.

WHAT IT MEASURES
    1. PROGRESS -- every progress event with its wall-clock offset and percent,
       so a frozen or saturating bar is visible in the data. Reports the largest
       gap between consecutive updates, which is what a user perceives as stuck.
    2. TIMESTAMP ACCURACY -- for phrases taken from the transcript itself at
       positions spread across the whole video, does the query return the
       timestamp those words actually occupy?
    3. SCORE DISTRIBUTION -- exact quotes, realistically imperfect quotes
       (punctuation, casing, a dropped word, an expanded contraction), and
       phrases that are definitely absent. The gap between the true-match and
       false-match distributions is what the thresholds have to sit inside.

    Nothing is downloaded: indexing fetches the audio stream only, exactly as
    the normal pipeline does.

USAGE
    python scripts/test_series.py --index      # build indexes (slow, once)
    python scripts/test_series.py --match      # threshold analysis (fast)
    python scripts/test_series.py --frames     # end-to-end incl. extraction
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, paths                                    # noqa: E402
from app.core.audio import ensure_audio, ensure_probe            # noqa: E402
from app.core.index import ensure_index, load_index              # noqa: E402
from app.core.matching import find_matches, format_timestamp     # noqa: E402
from app.core.resolve import resolve_cached                      # noqa: E402
from app.errors import Quest1Error                               # noqa: E402

# A spread of real videos: very short, and several minutes with continuous
# narration. Chosen so dialogue exists from the first seconds to the last.
TEST_VIDEOS = [
    ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "19s   first YouTube video"),
    ("https://www.youtube.com/watch?v=CxC161GvMPc", "298s  TED-Ed narration"),
    ("https://www.youtube.com/watch?v=Ke6XX8FHOHM", "465s  NASA mission briefing"),
    ("https://www.youtube.com/watch?v=cWs4WA--eKU", "530s  narrated short film"),
]

# Where in each video to sample a target phrase from, as a fraction of the word
# list. Deliberately includes both ends: a bug in the offset mapping or the
# frame clamp shows up at the extremes, not in the middle.
SAMPLE_POSITIONS = [0.02, 0.25, 0.50, 0.75, 0.97]

# How many consecutive words make up a sampled target phrase.
PHRASE_WORDS = 7

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "_test_series.json"


# --------------------------------------------------------------------------- #
# Progress capture
# --------------------------------------------------------------------------- #

class ProgressRecorder:
    """Records every progress event with the wall-clock time it arrived.

    USED BY: index_videos, to detect a bar that freezes or saturates.
    """

    def __init__(self) -> None:
        self.started = time.time()
        self.events: list[dict[str, Any]] = []

    def __call__(self, stage: str, percent: Optional[float], message: str) -> None:
        self.events.append({
            "t": round(time.time() - self.started, 2),
            "stage": stage,
            "percent": percent,
            "message": message,
        })

    def asr_gap_report(self) -> dict[str, Any]:
        """Largest silence between consecutive updates during the ASR stage."""
        asr = [e for e in self.events if e["stage"] == "asr"]
        if len(asr) < 2:
            return {"asr_events": len(asr), "max_gap": None, "percent_range": None}
        gaps = [b["t"] - a["t"] for a, b in zip(asr, asr[1:])]
        pcts = [e["percent"] for e in asr if e["percent"] is not None]
        return {
            "asr_events": len(asr),
            "max_gap": round(max(gaps), 1),
            "median_gap": round(statistics.median(gaps), 1),
            "percent_range": [round(min(pcts), 1), round(max(pcts), 1)] if pcts else None,
        }


# --------------------------------------------------------------------------- #
# Phase 1 -- indexing
# --------------------------------------------------------------------------- #

def index_videos(urls: list[tuple[str, str]]) -> dict[str, Any]:
    """Resolve, fetch audio and index each video, recording progress behaviour."""
    out: dict[str, Any] = {}
    for url, label in urls:
        print(f"\n{'=' * 78}\n{label}\n{url}\n{'=' * 78}", flush=True)
        recorder = ProgressRecorder()
        try:
            media, _ = resolve_cached(url, check_ranges=False, progress_callback=recorder)
            wav = ensure_audio(media, progress_callback=recorder)
            probe = ensure_probe(media, progress_callback=recorder)
            t0 = time.time()
            index = ensure_index(media.media_key, wav, progress_callback=recorder)
            asr_wall = time.time() - t0
        except Quest1Error as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            out[url] = {"label": label, "error": str(exc)}
            continue

        gaps = recorder.asr_gap_report()
        out[url] = {
            "label": label,
            "media_key": media.media_key,
            "title": media.title,
            "duration": media.duration,
            "fps": probe.get("fps"),
            "is_vfr": probe.get("is_vfr"),
            "words": index.word_count,
            "from_cache": index.from_cache,
            "asr_wall_seconds": round(asr_wall, 1),
            "realtime_factor": round(media.duration / asr_wall, 2) if asr_wall else None,
            "progress": gaps,
        }
        print(f"  {index.word_count} words, ASR wall {asr_wall:.1f}s, "
              f"progress: {gaps}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Phase 2 -- matching / threshold analysis
# --------------------------------------------------------------------------- #

def perturb(phrase: str) -> list[tuple[str, str]]:
    """Realistic ways a user's typed quote differs from the transcript.

    These are the cases the thresholds must tolerate: someone quoting from
    memory does not reproduce the recogniser's punctuation or every filler word.
    """
    words = phrase.split()
    variants = [("exact", phrase), ("caps+punct", phrase.upper() + "!!!")]
    if len(words) > 3:
        variants.append(("drop-middle", " ".join(words[:len(words) // 2]
                                                 + words[len(words) // 2 + 1:])))
        variants.append(("first-half", " ".join(words[:max(2, len(words) // 2)])))
    contracted = phrase.replace(" not", "n't").replace(" is", "'s")
    if contracted != phrase:
        variants.append(("contracted", contracted))
    return variants


# Phrases that appear in none of the test videos, to measure the false-match floor.
ABSENT_PHRASES = [
    "may the force be with you always",
    "frankly my dear I do not give a damn",
    "the quick brown fox jumps over the lazy dog",
    "supercalifragilisticexpialidocious nonsense words here",
]


def match_tests(indexed: dict[str, Any]) -> dict[str, Any]:
    """Query each index with sampled phrases and record scores + timing error."""
    true_scores: list[float] = []
    false_scores: list[float] = []
    rows: list[dict[str, Any]] = []

    for url, meta in indexed.items():
        if "error" in meta:
            continue
        index = load_index(meta["media_key"])
        if index is None:
            print(f"  no index for {meta['media_key']}", flush=True)
            continue

        print(f"\n--- {meta['label']}  ({index.word_count} words) ---", flush=True)

        for frac in SAMPLE_POSITIONS:
            start = min(int(index.word_count * frac), index.word_count - PHRASE_WORDS)
            start = max(0, start)
            picked = index.words[start:start + PHRASE_WORDS]
            if len(picked) < 3:
                continue
            phrase = " ".join(w["word"] for w in picked)
            expected_start = picked[0]["start"]

            for kind, variant in perturb(phrase):
                try:
                    result = find_matches(index, variant,
                                          progress_callback=lambda *_: None)
                except Quest1Error as exc:
                    rows.append({"video": meta["label"], "frac": frac, "kind": kind,
                                 "error": str(exc)})
                    continue

                best = result.best
                score = best.score if best else None
                err = abs(best.start_time - expected_start) if best else None
                if score is not None:
                    true_scores.append(score)
                rows.append({
                    "video": meta["label"], "frac": frac, "kind": kind,
                    "score": score, "band": result.band,
                    "expected_s": round(expected_start, 2),
                    "got_s": round(best.start_time, 2) if best else None,
                    "error_s": round(err, 2) if err is not None else None,
                    "occurrences": len(result.occurrences),
                })
                flag = "" if (err is not None and err < 0.75) else "   <-- CHECK"
                print(f"  {frac:>4.0%} {kind:<12} score={str(score):<6} "
                      f"band={result.band:<9} err={err if err is None else round(err,2)}s{flag}",
                      flush=True)

        # False-match floor: how high do unrelated phrases score on this video?
        for absent in ABSENT_PHRASES:
            result = find_matches(index, absent, progress_callback=lambda *_: None)
            top = (result.occurrences or result.near_misses)
            if top:
                false_scores.append(max(o.score for o in top))
            rows.append({
                "video": meta["label"], "frac": None, "kind": "absent",
                "phrase": absent, "band": result.band,
                "score": max((o.score for o in top), default=None),
                "occurrences": len(result.occurrences),
            })
            status = "OK" if not result.found else "FALSE POSITIVE"
            print(f"  absent: {status:<15} best={max((o.score for o in top), default=0):.1f} "
                  f"band={result.band}", flush=True)

    return {"rows": rows, "true_scores": true_scores, "false_scores": false_scores}


def summarise(analysis: dict[str, Any]) -> None:
    """Print the score distributions the thresholds have to separate."""
    true_s = sorted(analysis["true_scores"])
    false_s = sorted(analysis["false_scores"])
    print(f"\n{'=' * 78}\nTHRESHOLD ANALYSIS\n{'=' * 78}")

    def describe(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"{name}: no samples")
            return
        print(f"{name}: n={len(xs)}  min={min(xs):.1f}  p10={xs[len(xs)//10]:.1f}  "
              f"median={statistics.median(xs):.1f}  max={max(xs):.1f}")

    describe("TRUE  matches (sampled from transcript)", true_s)
    describe("FALSE matches (absent phrases)         ", false_s)

    if true_s and false_s:
        print(f"\ncurrent MATCH_THRESHOLD     = {config.MATCH_THRESHOLD}")
        print(f"current CONFIDENT_THRESHOLD = {config.CONFIDENT_THRESHOLD}")
        print(f"current NEAR_MISS_THRESHOLD = {config.NEAR_MISS_THRESHOLD}")
        print(f"\nhighest FALSE score  = {max(false_s):.1f}   <-- MATCH_THRESHOLD must exceed this")
        print(f"lowest  TRUE  score  = {min(true_s):.1f}   <-- MATCH_THRESHOLD must not exceed this")
        gap = min(true_s) - max(false_s)
        print(f"separation gap       = {gap:.1f}")
        if gap <= 0:
            print("  WARNING: the distributions OVERLAP -- no threshold separates them cleanly.")

    errors = [r["error_s"] for r in analysis["rows"]
              if r.get("error_s") is not None and r.get("kind") == "exact"]
    if errors:
        print(f"\nEXACT-quote timestamp error: n={len(errors)}  max={max(errors):.2f}s  "
              f"median={statistics.median(errors):.2f}s")
        bad = [r for r in analysis["rows"]
               if r.get("kind") == "exact" and (r.get("error_s") or 0) >= 0.75]
        print(f"  exact quotes landing >=0.75s away: {len(bad)}")
        for r in bad:
            print(f"    {r['video']} @ {r['frac']:.0%}: expected {r['expected_s']}s "
                  f"got {r['got_s']}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quest1 multi-video test series")
    parser.add_argument("--index", action="store_true", help="Build indexes (slow)")
    parser.add_argument("--match", action="store_true", help="Threshold analysis")
    parser.add_argument("--all", action="store_true", help="Both")
    args = parser.parse_args()
    if not (args.index or args.match or args.all):
        parser.error("choose --index, --match or --all")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}
    if RESULTS_PATH.exists():
        state = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    if args.index or args.all:
        state["indexed"] = index_videos(TEST_VIDEOS)
        RESULTS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

    if args.match or args.all:
        indexed = state.get("indexed") or {}
        if not indexed:
            print("No indexed videos. Run with --index first.", file=sys.stderr)
            return 2
        state["analysis"] = match_tests(indexed)
        summarise(state["analysis"])
        RESULTS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"\nresults written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
