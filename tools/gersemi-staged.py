#!/usr/bin/env python3
"""Format staged CMake hunks; full-file check when nothing is staged.

Per file: if it has staged hunks, reformat only those hunks in place via
``gersemi --line-ranges``; if it has none (CI ``pre-commit run --all-files`` on
a fresh clone), check the whole file with ``gersemi --check`` and fail on any
formatting diff.
"""

from __future__ import annotations

import subprocess
import sys

from staged_ranges import get_staged_line_ranges
from staged_ranges import repo_relative_paths


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        return 0

    ranges_by_path = get_staged_line_ranges(paths)
    modified = []
    to_check = []

    for path, relpath in zip(paths, repo_relative_paths(paths)):
        ranges = ranges_by_path.get(relpath, [])
        if not ranges:
            # Nothing staged for this file (the --all-files case): defer to a
            # whole-file check below.
            to_check.append(path)
            continue

        before = open(path, "rb").read()
        range_arg = ",".join(f"{start}-{end}" for start, end in ranges)
        result = subprocess.run(
            ["gersemi", "--in-place", "--line-ranges", range_arg, path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()
            if output:
                print(output, file=sys.stderr)
            return result.returncode

        after = open(path, "rb").read()
        if after != before:
            modified.append(path)

    rc = 0

    if to_check:
        result = subprocess.run(
            ["gersemi", "--check", *to_check],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()
            if output:
                print(output, file=sys.stderr)
            rc = 1

    if modified:
        for path in modified:
            print(f"reformatted staged CMake hunks in {path}")
        rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
