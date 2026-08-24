"""The ONE text normalization function, used at index time AND at query time.

If these two ever diverge, matching breaks in ways that look like bad ASR rather
than a bug, so every caller must go through `normalize_text`. Any change to the
rules below is a breaking change to every index on disk and MUST be accompanied
by a bump of config.INDEX_VERSION.

Rules, in order:
    1. Unicode NFKC  -- collapses ligatures and full-width forms.
    2. Curly quotes/dashes -> ASCII equivalents.
    3. Hyphens, slashes, underscores -> a space, so "well-known" matches the
       spoken "well known".
    4. Everything that is not a letter, digit, or space is dropped, so "don't"
       and "dont" normalize identically.
    5. Whitespace collapsed to single spaces, then stripped.

Run directly to see what a string normalizes to:
    python -m app.core.normalize "Don't -- you DARE!"
"""

from __future__ import annotations

import re
import sys
import unicodedata

# Characters that should become a space rather than vanish, so the words they
# join stay separate tokens.
_SPLIT_CHARS = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014\u2015_/\|]+")

# Anything left that is not a word character or whitespace is simply removed.
_STRIP_CHARS = re.compile(r"[^\w\s]", re.UNICODE)

_WHITESPACE = re.compile(r"\s+")

# Typographic characters NFKC does not fold on its own.
_TRANSLATIONS = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u00a0": " ", "\u200b": "", "\ufeff": "",
})


def normalize_text(text: str) -> str:
    """Fold `text` into the canonical form used by the index and by queries.

    THE most important function in the project for correctness. It runs in two
    places that MUST agree: once per word when building the index, and once per
    query at search time. If those ever diverged, a correct query would silently
    fail to match a correct transcript, and the bug would look like bad ASR.

    Returns "" for input that contains no matchable characters at all -- callers
    treat that as invalid input rather than as an empty search.

    USED BY: index.build_flat_text (index time), matching.find_matches (query
    time), and index.main for display.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"normalize_text expects a string, got {type(text).__name__}")

    folded = unicodedata.normalize("NFKC", text).translate(_TRANSLATIONS).lower()
    folded = _SPLIT_CHARS.sub(" ", folded)
    folded = _STRIP_CHARS.sub("", folded)
    return _WHITESPACE.sub(" ", folded).strip()


def normalize_tokens(text: str) -> list[str]:
    """Convenience: normalized text split into whitespace-separated tokens.

    WHY IT EXISTS: some callers want words rather than a flat string (counting
    query length, debugging). Built on normalize_text so it can never drift from
    the canonical rules.

    USED BY: the __main__ block below, and available to future callers that need
    token-level access.
    """
    normalized = normalize_text(text)
    return normalized.split(" ") if normalized else []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python -m app.core.normalize "<text>"', file=sys.stderr)
        raise SystemExit(2)
    for raw in sys.argv[1:]:
        print(f"{raw!r}\n  -> {normalize_text(raw)!r}\n  -> {normalize_tokens(raw)}")
