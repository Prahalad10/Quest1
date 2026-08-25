"""Run a set of dialogue searches under a hard per-case time limit and tabulate.

WHY THIS EXISTS
    "It works" and "it works inside the budget" are different claims. This
    harness measures the second one: every case gets a wall-clock deadline, and
    a case that overruns is KILLED and reported as an overrun rather than
    silently making the suite take an hour. That distinction is the whole point
    -- an aborted case is a result, not a missing row.

WHY A SUBPROCESS PER CASE
    A deadline can only be enforced on something that can be killed. Calling
    find_dialogue() in-process would give no way to stop a transcription that
    has run long. Running the CLI as a child, and killing the whole process TREE
    on timeout, also cleans up the parallel ASR workers -- killing only the
    parent would leave four decoders running and poison every later measurement.

Run:
    python -m scripts.test_matrix                 # every case, cold
    python -m scripts.test_matrix --warm          # reuse whatever is cached
    python -m scripts.test_matrix --limit 180     # change the deadline
    python -m scripts.test_matrix --only okru     # one case by label
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app import config, paths  # noqa: E402
from app.core.resolve import resolve  # noqa: E402


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
# Chosen to vary along the two axes that actually change the cost and the risk:
#   duration  -- 19s to feature length, because ASR cost is linear in duration
#   position  -- near the start, middle and end, because a match found early
#                proves nothing about whether the tail of the index is correct
#
# "expect" is the time the FIRST WORD OF THE QUERY is spoken, verified against a
# real run. It is not a guess. An earlier version took it from the start of a
# nine-word transcript window that began a few words before the phrase, which
# made correct results look 1.8-2.8s late -- the harness was wrong, not the
# pipeline. If a case here starts failing, check the anchor before the matcher.
CASES: list[dict[str, Any]] = [
    {"label": "yt-19s", "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
     "text": "they have really really long trunks", "expect": 7.38,
     "note": "19s clip -- smallest possible cold run"},
    {"label": "yt-13m-dub", "url": "https://www.youtube.com/watch?v=2LOh_01i8Is",
     "text": "that's my video now", "expect": 7.91,
     "note": "12.8m, 14 audio languages -- the dub-selection regression test"},
    {"label": "yt-13m-late", "url": "https://www.youtube.com/watch?v=2LOh_01i8Is",
     "text": "without risking the safety of our contestants", "expect": 367.36,
     "note": "same video, line at 48% -- tail of a multi-track index"},
    {"label": "yt-96m", "url": "https://www.youtube.com/watch?v=f7jkZXvaB4g",
     "text": "they're coming to get you Barbara", "expect": 367.83,
     "note": "Night of the Living Dead, 95.7m -- past the old 60m cap"},
    {"label": "okru-63m", "url": "https://ok.ru/video/9675994761971",
     "text": "water pours through your greatest dam", "expect": 189.86,
     "note": "ok.ru with a real audio-only DASH track"},
    {"label": "okru-54m", "url": "https://ok.ru/video/248244667877",
     "text": "My mind rebels at stagnation", "expect": 325.12,
     "note": "ok.ru with NO audio-only track -- progressive fallback"},
    {"label": "okru-absent", "url": "https://ok.ru/video/9675994761971",
     "text": "My mind rebels at stagnation", "expect": None,
     "expect_absent": True,
     "note": "line genuinely absent -- NO MATCH is the correct answer"},
]


# --------------------------------------------------------------------------- #
# Running one case
# --------------------------------------------------------------------------- #

def clear_cache(url: str) -> Optional[str]:
    """Delete every cached artifact for `url` so the next run is genuinely cold.

    WHY: a warm run answers in about a second because the transcript is already
    on disk. Timing that and calling it "a 90-minute video in 1.2s" would be a
    lie about the thing being measured. Cold timing is the honest number.

    The media_key is not derivable from the URL alone -- it needs the
    extractor, video id and title, which only a resolve provides. An earlier
    version guessed it from the URL, raised TypeError, swallowed it, and
    cleared nothing at all, so every "cold" run was silently warm.

    Returns the media_key it cleared, or None if the URL will not resolve --
    which the run itself then reports properly.
    """
    resolve_cache = paths.resolve_cache_path(url)
    try:
        media = resolve(url)
    except Exception:  # noqa: BLE001 - an unresolvable URL is the case's own problem
        resolve_cache.unlink(missing_ok=True)
        return None

    directory = config.DATA_DIR / media.media_key
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    resolve_cache.unlink(missing_ok=True)
    return media.media_key


def kill_tree(process: subprocess.Popen) -> None:
    """Kill a child and everything it spawned.

    WHY NOT process.kill(): the pipeline starts a pool of ASR worker processes.
    Killing only the parent orphans them, they keep burning every core, and the
    next case in the suite is measured on a machine that is already busy -- so
    one overrun would corrupt every result after it.

    USED BY: run_case, on deadline expiry.
    """
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True, check=False, timeout=30,
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        process.kill()
    except Exception:  # noqa: BLE001
        pass


def run_case(case: dict[str, str], limit: float) -> dict[str, Any]:
    """Execute one dialogue search with a deadline, returning a result row.

    outcome is one of:
        SUCCESS   a match was found and a frame was written
        NO MATCH  the pipeline ran fine but nothing cleared the threshold
        ERROR     the pipeline raised (unsupported host, network, bad input)
        OVERRUN   killed at the deadline -- the measurement itself is the finding

    USED BY: main().
    """
    command = [
        sys.executable, "-u", "-m", "app.cli",
        case["url"], case["text"], "--json", "--quiet",
    ]
    started = time.time()
    process = subprocess.Popen(
        command, cwd=str(REPO),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=limit)
        elapsed = time.time() - started
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        kill_tree(process)
        # Drain the pipes so the killed child does not leave a zombie.
        try:
            process.communicate(timeout=15)
        except Exception:  # noqa: BLE001
            pass
        return {**case, "outcome": "OVERRUN", "elapsed": elapsed,
                "detail": f"killed at the {limit:.0f}s limit"}

    row: dict[str, Any] = {**case, "elapsed": elapsed}
    payload: Optional[dict[str, Any]] = None
    if stdout.strip():
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None

    if payload is None:
        row["outcome"] = "ERROR"
        row["detail"] = (stderr.strip().splitlines() or ["no output"])[-1][:160]
        return row

    status = payload.get("status")
    if status == "found":
        image = payload.get("image_path") or ""
        exists = Path(image).exists() if image else False
        seconds = payload.get("timestamp_seconds")
        expect = case.get("expect")
        # A match at the wrong place is not a success, so when the true time is
        # known it is checked. The slack below covers the matcher anchoring on
        # the second word of a phrase rather than the first -- a sub-second
        # difference, not a different line.
        off_by = abs(seconds - expect) if (expect is not None and seconds is not None) else None
        if not exists:
            row["outcome"], row["detail"] = "ERROR", "match found but frame image missing"
        elif off_by is not None and off_by > 1.5:
            row["outcome"] = "WRONG SPOT"
            row["detail"] = f"landed {off_by:.1f}s from the expected {expect:.1f}s"
        else:
            row["outcome"] = "SUCCESS"
            row["detail"] = "" if off_by is None else f"{off_by:.2f}s from expected"
        row["timestamp"] = payload.get("timestamp")
        row["frame"] = payload.get("frame_number")
        row["matched"] = payload.get("matched_text")
        row["score"] = payload.get("score")
        row["band"] = payload.get("band")
        row["image"] = image
    elif status == "not_found":
        # For a case whose line is genuinely absent, NO MATCH is the right answer.
        row["outcome"] = "NO MATCH (expected)" if case.get("expect_absent") else "NO MATCH"
        near = payload.get("near_misses") or []
        row["detail"] = (f"closest: {near[0]['matched_text'][:60]!r} "
                         f"(score {near[0]['score']})") if near else "nothing close"
    else:
        row["outcome"] = "ERROR"
        row["detail"] = str(payload.get("error", ""))[:160]
    return row


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_table(rows: list[dict[str, Any]], limit: float) -> None:
    """Print the results as a fixed-width table plus per-case detail.

    USED BY: main(). Kept separate so the rows can also be dumped as JSON.
    """
    headers = ("CASE", "TIME", "OUTCOME", "TIMESTAMP", "FRAME", "SCORE")
    widths = (12, 9, 8, 13, 8, 6)
    print()
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        cells = (
            str(row["label"])[:12],
            f"{row['elapsed']:.1f}s",
            row["outcome"],
            str(row.get("timestamp") or "-"),
            str(row.get("frame") if row.get("frame") is not None else "-"),
            f"{row['score']:.1f}" if row.get("score") is not None else "-",
        )
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    print()
    for row in rows:
        if row.get("detail"):
            print(f"  {row['label']}: {row['detail']}")
    print()
    ok = sum(1 for r in rows if r["outcome"].startswith(("SUCCESS", "NO MATCH (expected)")))
    over = sum(1 for r in rows if r["outcome"] == "OVERRUN")
    print(f"  {ok}/{len(rows)} SUCCESS, {over} overran the {limit:.0f}s limit")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point: run every selected case and print the table.

    USED BY: `python -m scripts.test_matrix`.
    """
    parser = argparse.ArgumentParser(prog="python -m scripts.test_matrix")
    parser.add_argument("--limit", type=float, default=180.0,
                        help="Per-case deadline in seconds (default 180)")
    parser.add_argument("--warm", action="store_true",
                        help="Do NOT clear caches first")
    parser.add_argument("--only", default=None,
                        help="Run just the case with this label")
    parser.add_argument("--json-out", default=None,
                        help="Also write the rows to this JSON file")
    args = parser.parse_args(argv)

    cases = [c for c in CASES if args.only is None or c["label"] == args.only]
    if not cases:
        print(f"No case labelled {args.only!r}. Known: "
              f"{', '.join(c['label'] for c in CASES)}", file=sys.stderr)
        return 2

    print(f"limit {args.limit:.0f}s per case, "
          f"{'warm (caches kept)' if args.warm else 'cold (caches cleared)'}, "
          f"model={config.ASR_MODEL} beam={config.ASR_BEAM_SIZE} "
          f"index_v{config.INDEX_VERSION}")

    rows: list[dict[str, Any]] = []
    for case in cases:
        if not args.warm:
            clear_cache(case["url"])
        print(f"\n>>> {case['label']}: {case['note']}", flush=True)
        row = run_case(case, args.limit)
        print(f"    {row['outcome']} in {row['elapsed']:.1f}s "
              f"{row.get('detail', '')}", flush=True)
        rows.append(row)

    print_table(rows, args.limit)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
