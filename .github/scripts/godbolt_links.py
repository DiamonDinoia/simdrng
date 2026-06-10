#!/usr/bin/env python3
"""
Generate Compiler Explorer shortlinks for the simdrng single-header demos.

Compiler Explorer cannot #include arbitrary URLs, so each link embeds the
amalgamated single header (the demo's own `#include "simdrng*.hpp"` line is
stripped and the header text is prepended). The SIMD demo additionally pulls in
Compiler Explorer's vendored xsimd library; poet is already inlined into the
SIMD header, so xsimd is the only external dependency.

Reads the generated headers from out/ and the demos from .github/godbolt/,
writes index.md + links.json to /tmp/godbolt.
"""

import json
import pathlib
import re
import sys
import urllib.request

OUT = pathlib.Path("out")
GODBOLT = pathlib.Path(".github/godbolt")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"simdrng[^"]*"\s*$', re.MULTILINE)
COMPILER_ID = "g142"  # gcc 14.2

DEMOS = [
    {
        "name": "scalar",
        "demo": GODBOLT / "demo_scalar.cpp",
        "header": OUT / "simdrng-scalar.hpp",
        "options": "-std=c++20 -O3",
        "libs": [],
    },
    {
        "name": "simd",
        "demo": GODBOLT / "demo_simd.cpp",
        "header": OUT / "simdrng.hpp",
        # AVX2 baseline runs on Compiler Explorer; xsimd "trunk" matches our
        # upstream-master pin (the version that has mul_hilo).
        "options": "-std=c++20 -O3 -march=x86-64-v3",
        "libs": [{"id": "xsimd", "version": "trunk"}],
    },
]


def shorten(source: str, options: str, libs: list) -> str:
    state = {
        "sessions": [{
            "id": 1,
            "language": "c++",
            "source": source,
            "compilers": [{"id": COMPILER_ID, "options": options, "libs": libs}],
        }]
    }
    req = urllib.request.Request(
        "https://godbolt.org/api/shortener",
        data=json.dumps(state).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["url"]


def main() -> int:
    links = {}
    failed = False
    for d in DEMOS:
        header = d["header"].read_text(encoding="utf-8")
        demo = INCLUDE_RE.sub("", d["demo"].read_text(encoding="utf-8"))
        source = f"{header}\n{demo}"
        try:
            links[d["name"]] = shorten(source, d["options"], d["libs"])
            print(f"{d['name']}: {links[d['name']]}")
        except Exception as e:  # noqa: BLE001 - best effort, report and continue
            print(f"WARN: shortlink failed for {d['name']}: {e}", file=sys.stderr)
            failed = True

    out = pathlib.Path("/tmp/godbolt")
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Compiler Explorer links", "",
             "Self-contained single-header demos (the SIMD one uses Compiler "
             "Explorer's vendored xsimd library):", ""]
    lines += [f"- [{name}]({url})" for name, url in sorted(links.items())]
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "links.json").write_text(json.dumps(links, indent=2) + "\n", encoding="utf-8")

    # Don't fail the whole run on a transient shortener hiccup, but do signal it.
    return 1 if failed and not links else 0


if __name__ == "__main__":
    sys.exit(main())
