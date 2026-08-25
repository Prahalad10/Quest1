#!/usr/bin/env python
"""Environment preflight: Python version, ffmpeg binaries, third-party imports.

Stdlib only -- this must run BEFORE `pip install -r requirements.txt`. Exits
non-zero if anything is missing, so it works as a gate in a script.

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys

MIN_PYTHON = (3, 11)
REQUIRED_IMPORTS = [("yt_dlp", "yt-dlp"), ("faster_whisper", "faster-whisper"),
                    ("rapidfuzz", "rapidfuzz")]
REQUIRED_BINARIES = ["ffmpeg", "ffprobe"]

# Printed, never executed -- the user approves and runs it themselves.
FFMPEG_INSTALL = {
    "Windows": "winget install --id=Gyan.FFmpeg -e --source winget",
    "Darwin": "brew install ffmpeg",
    "Linux": "sudo apt update && sudo apt install -y ffmpeg",
}


def check_python() -> tuple[str, bool, str]:
    ok = sys.version_info >= MIN_PYTHON
    return ("python >= %d.%d" % MIN_PYTHON, ok, platform.python_version())


def check_venv() -> tuple[str, bool, str]:
    """Installing into the system interpreter is the most common setup mistake."""
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return ("virtualenv active", active,
            sys.prefix if active else "not in a venv -- run: python -m venv venv")


def check_binary(name: str) -> tuple[str, bool, str]:
    """ffmpeg is a system dependency pip cannot install."""
    path = shutil.which(name)
    if path is None:
        return (name, False, "not found on PATH")
    try:
        out = subprocess.run([path, "-version"], capture_output=True, text=True,
                             timeout=15, check=False).stdout.splitlines()
        return (name, True, out[0] if out else path)
    except (OSError, subprocess.SubprocessError) as exc:
        return (name, False, f"found at {path} but would not run: {exc}")


def check_import(module: str, package: str) -> tuple[str, bool, str]:
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        return (f"import {module}", False, f"{exc} -- pip install {package}")
    return (f"import {module}", True, getattr(mod, "__version__", "installed"))


def main() -> int:
    rows = [check_venv(), check_python()]
    rows += [check_binary(b) for b in REQUIRED_BINARIES]
    rows += [check_import(m, p) for m, p in REQUIRED_IMPORTS]

    width = max(len(name) for name, _, _ in rows)
    for name, ok, detail in rows:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}    {detail}")

    failed = [name for name, ok, _ in rows if not ok]
    if any(b in failed for b in REQUIRED_BINARIES):
        hint = FFMPEG_INSTALL.get(platform.system())
        if hint:
            print(f"\n--- ffmpeg install command for {platform.system()} (NOT run) ---\n    {hint}")

    print(f"\nRESULT: {'PASS -- all %d checks passed.' % len(rows) if not failed else 'FAIL -- ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
