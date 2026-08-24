"""Command-line front end: find the video frame where a line of dialogue is spoken.

    python -m app.cli <video_url> "<dialogue text>"

WHY THIS FILE IS THIN
    All orchestration lives in app/service.py so that the CLI and the web API
    run identical logic. This module does three things and nothing else:
    parse arguments, call `find_dialogue`, and format the result for a terminal.

    If you are looking for how the pipeline actually works, read
    app/service.py -- not this file.

OUTPUT CONTRACT
    The required block is printed first, verbatim, in this order:

        Timestamp : HH:MM:SS.sss
        Frame     : <frame number>
        Text      : "<matched text>"
        Image     : <path to extracted frame PNG>

    Anything that merely qualifies the answer (score, confidence band,
    surrounding context, other occurrences) is printed after it, indented, so
    the required four lines stay easy to read and easy to parse.

EXIT CODES
    0  a match was found
    1  no match above threshold (near misses printed, if any)
    2  an error -- bad input, unsupported media, or a processing failure

    Distinct codes matter so a shell script can tell "this video does not
    contain that line" apart from "something broke".
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from app.core.matching import format_timestamp
from app.errors import Quest1Error
from app.progress import ProgressCallback, null_progress
from app.service import DialogueResult, find_dialogue


def print_result(result: DialogueResult) -> None:
    """Render a DialogueResult for a human reading a terminal.

    WHY SEPARATE FROM find_dialogue: the pipeline returns data, and formatting
    is a presentation concern. app/api.py serialises the same object to JSON
    without going anywhere near this function.

    USED BY: `main`, when --json was not passed.
    """
    print()
    if result.status == "found":
        # A variable-frame-rate source genuinely has no computable frame number.
        # Say so rather than printing a plausible-looking integer.
        frame_display = (
            str(result.frame_number) if result.frame_number is not None
            else "null (variable frame rate)"
        )
        print(f"Timestamp : {result.timestamp}")
        print(f"Frame     : {frame_display}")
        print(f'Text      : "{result.matched_text}"')
        print(f"Image     : {result.image_path}")
        print()
        print(f"  score     : {result.score}  ({result.band})")
        print(f"  context   : ...{result.context}...")
        if result.frame_note:
            print(f"  note      : {result.frame_note}")
        if result.other_occurrences:
            # Shown because an "ambiguous" band usually means the line is said
            # more than once, and only the user can say which one they meant.
            print("  also said at:")
            for occ in result.other_occurrences:
                print(f"    {format_timestamp(occ['start_time'])}  "
                      f"score {occ['score']}  {occ['matched_text']!r}")
    else:
        print(f"NOT FOUND : no dialogue matching {result.query!r} above threshold.")
        print()
        if result.near_misses:
            # The whole point of near misses: show what the transcript really
            # says so the user can tell a mis-typed quote from a mis-heard one.
            print("  closest lines in the transcript:")
            for occ in result.near_misses:
                print(f"    {format_timestamp(occ['start_time'])}  "
                      f"score {occ['score']}  {occ['matched_text']!r}")
        else:
            print("  Nothing in the transcript was close enough to suggest.")

    print()
    print(f"  media_key : {result.media_key}")
    print(f"  cached    : "
          f"{', '.join(result.cached_stages) if result.cached_stages else 'nothing (first run)'}")
    print(f"  elapsed   : {result.elapsed_seconds:.2f}s")
    print()


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface.

    WHY A SEPARATE FUNCTION: keeps `main` readable, and lets a test build the
    parser to assert on flags without executing anything.

    USED BY: `main`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Find the video frame where a line of dialogue is spoken.",
    )
    parser.add_argument("url", help="Video URL (a single video, not a playlist)")
    parser.add_argument("text", help="The line of dialogue to locate")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable output on stdout")
    parser.add_argument("--force", action="store_true",
                        help="Ignore every cache and recompute from scratch")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output on stderr")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point. Returns the process exit code rather than calling sys.exit.

    WHY RETURN INSTEAD OF EXIT: makes the CLI callable from a test or another
    Python process without terminating the interpreter.

    USED BY: the `if __name__ == "__main__"` block below, and by anything that
    wants to invoke the CLI programmatically.
    """
    args = build_parser().parse_args(argv)

    # None means "use the stderr default"; null_progress discards everything.
    progress: Optional[ProgressCallback] = null_progress if args.quiet else None

    try:
        result = find_dialogue(
            args.url, args.text, force=args.force, progress_callback=progress,
        )
    except Quest1Error as exc:
        # Every Quest1Error carries a message written to be shown to a user, so
        # print it plainly. A stack trace here would be noise, not information.
        if args.json:
            print(json.dumps({
                "status": "error",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }, indent=2, ensure_ascii=False))
        else:
            print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print_result(result)

    return 0 if result.status == "found" else 1


if __name__ == "__main__":
    sys.exit(main())
