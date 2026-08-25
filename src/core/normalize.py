"""The ONE text normalization, used at index time and at query time.

If those two ever diverge, matching fails in ways that look like bad ASR. Any
change to these rules invalidates every index on disk and must bump
config.INDEX_VERSION.

NFKC -> fold typographic quotes -> hyphen/slash/underscore to space (so
"well-known" matches spoken "well known") -> drop non-alphanumerics (so "don't"
== "dont") -> collapse whitespace.

    python -m src.core.normalize "Don't -- you DARE!"
"""

from __future__ import annotations

import re
import sys
import unicodedata

_SPLIT_CHARS = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014\u2015_/\|]+")
_STRIP_CHARS = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_TRANSLATIONS = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u00a0": " ", "\u200b": "", "\ufeff": "",
})


def normalize_text(text: str) -> str:
    """Fold text into the canonical form. Returns "" if nothing matchable remains."""
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"normalize_text expects a string, got {type(text).__name__}")
    folded = unicodedata.normalize("NFKC", text).translate(_TRANSLATIONS).lower()
    folded = _SPLIT_CHARS.sub(" ", folded)
    folded = _STRIP_CHARS.sub("", folded)
    return _WHITESPACE.sub(" ", folded).strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python -m src.core.normalize "<text>"', file=sys.stderr)
        raise SystemExit(2)
    for raw in sys.argv[1:]:
        print(f"{raw!r} -> {normalize_text(raw)!r}")
