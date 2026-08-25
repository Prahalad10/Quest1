"""Terminal front end -- a thin wrapper over service.find_dialogue.

    python -m src.cli <video_url> "<dialogue text>"

Exit codes are distinct so a script can tell "not in this video" from "broken":
    0 found | 1 no match (near misses printed) | 2 error
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from src.core.matching import format_timestamp
from src.errors import DialogueFrameError
from src.progress import ProgressCallback, null_progress
from src.service import DialogueResult, find_dialogue


def print_result(result: DialogueResult) -> None:
    """The four required lines, then the detail needed to judge the answer."""
    print()
    if result.status == "found":
        frame = (str(result.frame_number) if result.frame_number is not None
                 else "null (variable frame rate)")
        print(f"Timestamp : {result.timestamp}")
        print(f"Frame     : {frame}")
        print(f'Text      : "{result.matched_text}"')
        print(f"Image     : {result.image_path}")
        print()
        print(f"  score     : {result.score}  ({result.band})")
        print(f"  context   : ...{result.context}...")
        if result.frame_note:
            print(f"  note      : {result.frame_note}")
        if result.other_occurrences:
            # An "ambiguous" band usually means the line is said more than once,
            # and only the user can say which one they meant.
            print("  also said at:")
            for occ in result.other_occurrences:
                print(f"    {format_timestamp(occ['start_time'])}  "
                      f"score {occ['score']}  {occ['matched_text']!r}")
    else:
        print(f"NOT FOUND : no dialogue matching {result.query!r} above threshold.")
        print()
        if result.near_misses:
            # Show what the transcript really says, so the user can tell a
            # mistyped quote from a misheard one.
            print("  closest lines in the transcript:")
            for occ in result.near_misses:
                print(f"    {format_timestamp(occ['start_time'])}  "
                      f"score {occ['score']}  {occ['matched_text']!r}")
        else:
            print("  Nothing in the transcript was close enough to suggest.")

    print()
    print(f"  media_key : {result.media_key}")
    print(f"  cached    : {', '.join(result.cached_stages) or 'nothing (first run)'}")
    print(f"  elapsed   : {result.elapsed_seconds:.2f}s")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    """Returns the exit code rather than calling sys.exit, so it stays callable
    from another process or a test."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Find the video frame where a line of dialogue is spoken.",
    )
    parser.add_argument("url", help="Video URL (a single video, not a playlist)")
    parser.add_argument("text", help="The line of dialogue to locate")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument("--force", action="store_true", help="Ignore every cache")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress on stderr")
    args = parser.parse_args(argv)

    # None means "use the stderr default"; null_progress discards.
    progress: Optional[ProgressCallback] = null_progress if args.quiet else None

    try:
        result = find_dialogue(args.url, args.text, force=args.force, progress_callback=progress)
    except DialogueFrameError as exc:
        # Every DialogueFrameError message is written to be shown to a user, so print
        # it plainly -- a stack trace here would be noise, not information.
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc),
                              "error_type": type(exc).__name__}, indent=2, ensure_ascii=False))
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
