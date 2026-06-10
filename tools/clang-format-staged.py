#!/usr/bin/env python3
"""Format staged C/C++ hunks; full-file check when nothing is staged.

Two modes, chosen by whether the passed files have staged hunks:

* Staged hunks present (normal ``git commit``): only the staged hunks are
  reformatted via ``git-clang-format --staged``. git-clang-format uses a fixed
  temp-index path (.git/clang-format-index) rather than a unique temp file; when
  pre-commit's concurrent git operations race with it the .lock file persists
  and blocks subsequent commits, so we clean stale locks around the run.
  See: https://github.com/llvm/llvm-project/issues/52644

* Nothing staged (CI ``pre-commit run --all-files`` on a fresh clone): there are
  no staged hunks to map, so each file is checked whole with
  ``clang-format --dry-run --Werror`` and any formatting diff fails the hook.
"""

import os
import subprocess
import sys

from staged_ranges import get_staged_line_ranges


def git_dir():
    return subprocess.check_output(["git", "rev-parse", "--git-dir"], text=True).strip()


def remove_lock(lockpath):
    try:
        os.remove(lockpath)
    except FileNotFoundError:
        pass


def format_staged_hunks(paths):
    """Incremental path: reformat only staged hunks. Returns an exit code."""
    lock = os.path.join(git_dir(), "clang-format-index.lock")

    # Remove stale lock from a previous interrupted run.
    remove_lock(lock)

    cmd = ["git-clang-format", "--binary", "clang-format", "--staged", "--"]
    cmd.extend(paths)
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Clean up lock in case it leaked.
    remove_lock(lock)

    output = (result.stdout + result.stderr).strip()

    if result.returncode != 0:
        if output:
            print(output, file=sys.stderr)
        return result.returncode

    if "did not modify any files" in output or "no modified files" in output:
        return 0

    # Files were modified — print what changed and fail so pre-commit re-stages.
    if output:
        print(output)
    return 1


def check_whole_files(paths):
    """Full-file path: fail on any formatting diff. Returns an exit code."""
    rc = 0
    for path in paths:
        result = subprocess.run(
            ["clang-format", "--dry-run", "--Werror", path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()
            if output:
                print(output, file=sys.stderr)
            rc = 1
    return rc


def main():
    paths = sys.argv[1:]
    if not paths:
        return 0

    ranges_by_path = get_staged_line_ranges(paths)
    if any(ranges_by_path.values()):
        return format_staged_hunks(paths)
    return check_whole_files(paths)


if __name__ == "__main__":
    sys.exit(main())
