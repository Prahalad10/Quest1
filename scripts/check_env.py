#!/usr/bin/env python
"""Environment preflight check for Quest1.

Verifies the Python version, the ffmpeg/ffprobe binaries, and every third-party
import the pipeline depends on. Prints a pass/fail table and exits non-zero if
anything is missing, so it is safe to use as a gate in a script or CI step.

Stdlib only -- this must run *before* `pip install -r requirements.txt`.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys

MIN_PYTHON = (3, 11)

# (import name, pip name) for every third-party module the pipeline imports.
REQUIRED_IMPORTS = [
    ("yt_dlp", "yt-dlp"),
    ("faster_whisper", "faster-whisper"),
    ("rapidfuzz", "rapidfuzz"),
]

REQUIRED_BINARIES = ["ffmpeg", "ffprobe"]

# Printed, never executed. The user approves and runs it themselves.
FFMPEG_INSTALL_COMMANDS = {
    "Windows": "winget install --id=Gyan.FFmpeg -e --source winget",
    "Darwin": "brew install ffmpeg",
    "Linux": "sudo apt update && sudo apt install -y ffmpeg",
}


class Check:
    """One row of the results table: a name, a pass/fail, and an explanation.

    USED BY: every check_* function below, and render_table.
    """

    def __init__(self, name: str, ok: bool, detail: str) -> None:
        """Store one check outcome. `detail` is shown verbatim in the table."""
        self.name = name
        self.ok = ok
        self.detail = detail


def check_python() -> Check:
    """Verify the interpreter is new enough, and report WHICH interpreter it is.

    Printing sys.executable matters more than the version: the most common
    setup mistake is running the system Python while believing the venv is
    active, and the path makes that instantly visible.

    USED BY: main().
    """
    v = sys.version_info
    actual = f"{v.major}.{v.minor}.{v.micro}"
    ok = (v.major, v.minor) >= MIN_PYTHON
    want = ".".join(str(p) for p in MIN_PYTHON)
    detail = f"{actual}  ({sys.executable})" if ok else f"{actual} -- need >= {want}"
    return Check(f"python >= {want}", ok, detail)


def check_venv() -> Check:
    """Warn when running outside a virtualenv.

    Not a correctness problem on its own, but installing this project's
    dependencies globally is almost always a mistake, and faster-whisper pulls
    in several large packages.

    USED BY: main().
    """
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    detail = sys.prefix if in_venv else "NOT in a venv -- packages would install globally"
    return Check("virtualenv active", in_venv, detail)


def check_binary(name: str) -> Check:
    """Verify an external binary exists AND actually runs.

    WHY NOT JUST shutil.which: a file can be on PATH and still fail to execute --
    a broken symlink, a wrong architecture, or a Windows execution alias that
    resolves but errors. Running `-version` proves it works.

    USED BY: main(), for ffmpeg and ffprobe.
    """
    path = shutil.which(name)
    if path is None:
        return Check(name, False, "not found on PATH")
    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check(name, False, f"found at {path} but failed to run: {exc}")
    if proc.returncode != 0:
        return Check(name, False, f"{path} exited {proc.returncode}")
    first_line = (proc.stdout or proc.stderr).strip().splitlines()[0]
    return Check(name, True, first_line)


def check_import(module_name: str, pip_name: str) -> Check:
    """Verify a dependency imports, reporting its version.

    Catches ImportError (not installed) separately from other exceptions (a
    broken or half-installed package), because those need different fixes and a
    broken install must never be reported as merely missing.

    USED BY: main(), once per entry in REQUIRED_IMPORTS.
    """
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        return Check(f"import {module_name}", False, f"missing -- pip install {pip_name} ({exc})")
    except Exception as exc:  # noqa: BLE001 - a broken install must be reported, not hidden
        return Check(f"import {module_name}", False, f"import raised {type(exc).__name__}: {exc}")
    version = getattr(mod, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version

            version = dist_version(pip_name)
        except Exception:  # noqa: BLE001 - version is informational only
            version = "unknown version"
    return Check(f"import {module_name}", True, str(version))


def render_table(checks: list[Check]) -> None:
    """Print all checks as an aligned PASS/FAIL table.

    USED BY: main().
    """
    name_w = max(len(c.name) for c in checks)
    print()
    print(f"{'CHECK'.ljust(name_w)}  {'RESULT':6}  DETAIL")
    print(f"{'-' * name_w}  {'-' * 6}  {'-' * 60}")
    for c in checks:
        print(f"{c.name.ljust(name_w)}  {'PASS' if c.ok else 'FAIL':6}  {c.detail}")
    print()


def print_hints(checks: list[Check]) -> None:
    """Print how to fix whatever failed, plus the ffmpeg command for this OS.

    The ffmpeg command is PRINTED, NEVER RUN. Installing a system package is a
    change to the user machine and is theirs to approve -- this script only ever
    reports.

    USED BY: main().
    """
    failed = {c.name for c in checks if not c.ok}
    system = platform.system()

    print("--- ffmpeg install command for this OS "
          f"({system or 'unknown'}) -- NOT run by this script ---")
    print(f"    {FFMPEG_INSTALL_COMMANDS.get(system, 'see https://ffmpeg.org/download.html')}")
    print("    (installs both ffmpeg and ffprobe; reopen the terminal afterwards)")
    print()

    missing_imports = [n for n in failed if n.startswith("import ")]
    if missing_imports:
        print("--- missing Python packages ---")
        print("    pip install -r requirements.txt")
        print()

    if "virtualenv active" in failed:
        print("--- no virtualenv ---")
        print("    Create and activate one before installing (see project setup commands).")
        print()


def main() -> int:
    """Run every check, print the table and hints, return a shell exit code.

    Returns 1 if anything failed, so this can gate a setup script or CI step.

    USED BY: `python scripts/check_env.py`.
    """
    checks: list[Check] = [check_python(), check_venv()]
    checks += [check_binary(b) for b in REQUIRED_BINARIES]
    checks += [check_import(mod, pip) for mod, pip in REQUIRED_IMPORTS]

    render_table(checks)
    print_hints(checks)

    failures = [c for c in checks if not c.ok]
    if failures:
        print(f"RESULT: FAIL -- {len(failures)} of {len(checks)} checks failed: "
              f"{', '.join(c.name for c in failures)}")
        return 1
    print(f"RESULT: PASS -- all {len(checks)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
