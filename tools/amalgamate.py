#!/usr/bin/env python3
"""
Amalgamate the simdrng headers into a single self-contained header.

Inlines project headers (resolved under include/, plus any --extra-root) reachable
from the entry header, strips their include guards / `#pragma once`, and preserves
other system includes (e.g. <xsimd/xsimd.hpp>, <cstdint>) verbatim.

Two variants are produced from the same source tree:

  * SIMD (default): inline poet via --extra-root so the only external dependency
    is xsimd (kept as <xsimd/...>; provided by the system or the Compiler Explorer
    xsimd library). Pass --xsimd-default to default SIMDRNG_WITH_XSIMD to 1.

  * scalar (--strip-xsimd): evaluate `#if SIMDRNG_WITH_XSIMD` as 0 and drop those
    blocks (keeping any #else), and bake `#define SIMDRNG_WITH_XSIMD 0`. The result
    is fully self-contained: no xsimd, no poet, no project include path.

Usage:
  tools/amalgamate.py [--root ROOT] [--input PATH] [--output PATH] [--guard MACRO]
                      [--extra-root DIR ...] [--strip-xsimd] [--xsimd-default]
"""

from pathlib import Path
import argparse
import re
import sys

INCLUDE_Q_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"\s*(?://.*)?$')
INCLUDE_LT_RE = re.compile(r"^\s*#\s*include\s*<([^>]+)>\s*(?://.*)?$")
IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\s+([A-Z0-9_]+)\s*$")
DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Z0-9_]+)\s*$")
ENDIF_RE = re.compile(r"^\s*#\s*endif\b")
PRAGMA_ONCE_RE = re.compile(r"^\s*#\s*pragma\s+once\b")

# Conditional directives, for the minimal preprocessor used by --strip-xsimd.
XSIMD_IF_RE = re.compile(r"^\s*#\s*if\s+SIMDRNG_WITH_XSIMD\s*$")
ANY_IF_RE = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef)\b")
ELSE_RE = re.compile(r"^\s*#\s*else\b")
ELIF_RE = re.compile(r"^\s*#\s*elif\b")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def is_under(path: Path, roots) -> bool:
    for r in roots:
        try:
            path.relative_to(r)
            return True
        except ValueError:
            continue
    return False


def strip_header_guard(text: str) -> str:
    """Remove a leading #ifndef/#define guard + trailing #endif, and #pragma once."""
    lines = [ln for ln in text.splitlines() if not PRAGMA_ONCE_RE.match(ln)]
    guard = None
    guard_idx = None
    for i in range(min(8, max(0, len(lines) - 1))):
        m1 = IFNDEF_RE.match(lines[i])
        if m1:
            m2 = DEFINE_RE.match(lines[i + 1]) if i + 1 < len(lines) else None
            if m2 and m1.group(1) == m2.group(1):
                guard = m1.group(1)
                guard_idx = i
                break
    if guard and guard_idx is not None:
        del lines[guard_idx : guard_idx + 2]
        for j in range(len(lines) - 1, -1, -1):
            if ENDIF_RE.match(lines[j]):
                del lines[j]
                break
    return "\n".join(lines) + "\n"


def resolve_inlinable(name: str, current_file: Path, search_dirs, allowed_roots):
    """Return the path of an inlinable project/vendored header, or None to keep verbatim."""
    for d in [current_file.parent] + list(search_dirs):
        p = (d / name).resolve()
        if p.exists() and is_under(p, allowed_roots):
            return p
    return None


def inline_file(path: Path, processed: set, search_dirs, allowed_roots, strip_xsimd: bool) -> str:
    path = path.resolve()
    if str(path) in processed:
        return f"/* Skipped already inlined: {path.name} */\n"
    processed.add(str(path))

    text = strip_header_guard(read_text(path))
    out = [f"// BEGIN_FILE: {path.name}\n"]

    # Minimal per-file preprocessor for --strip-xsimd: a stack of conditional
    # frames. 'eval' frames are `#if SIMDRNG_WITH_XSIMD` (value forced to 0);
    # 'opaque' frames are every other #if/#ifdef/#ifndef (passed through as text
    # when the surrounding eval frames are active). Conditionals are balanced
    # within a single file, so a per-file stack is sufficient.
    stack = []

    def emitting() -> bool:
        return all(f["active"] for f in stack if f["kind"] == "eval")

    for line in text.splitlines():
        if strip_xsimd:
            if XSIMD_IF_RE.match(line):
                parent = emitting()
                stack.append({"kind": "eval", "active": False, "parent": parent})
                continue  # SIMDRNG_WITH_XSIMD == 0: the if-branch is dead
            if ELSE_RE.match(line) and stack and stack[-1]["kind"] == "eval":
                fr = stack[-1]
                fr["active"] = fr["parent"]  # take the #else branch
                continue
            if ENDIF_RE.match(line) and stack and stack[-1]["kind"] == "eval":
                stack.pop()
                continue
            if ANY_IF_RE.match(line):
                if emitting():
                    out.append(line + "\n")
                stack.append({"kind": "opaque", "active": True})
                continue
            if ELSE_RE.match(line) or ELIF_RE.match(line):
                if emitting():
                    out.append(line + "\n")
                continue
            if ENDIF_RE.match(line):
                if stack and stack[-1]["kind"] == "opaque":
                    stack.pop()
                if emitting():
                    out.append(line + "\n")
                continue
            if not emitting():
                continue

        matched = False
        for regex, kind in ((INCLUDE_Q_RE, "quoted"), (INCLUDE_LT_RE, "angle")):
            m = regex.match(line)
            if m:
                inc = resolve_inlinable(m.group(1), path, search_dirs, allowed_roots)
                if inc:
                    out.append(f"/* Begin inline ({kind}): {inc.name} */\n")
                    out.append(inline_file(inc, processed, search_dirs, allowed_roots, strip_xsimd))
                    out.append(f"/* End inline ({kind}): {inc.name} */\n")
                else:
                    out.append(line + "\n")  # external header: keep verbatim
                matched = True
                break
        if not matched:
            out.append(line + "\n")

    out.append(f"// END_FILE: {path.name}\n")
    return "".join(out)


def make_banner(strip_xsimd: bool) -> str:
    if strip_xsimd:
        return (
            "/* Auto-generated single-header for simdrng (scalar variant).\n"
            " * Do not edit directly.\n"
            " *\n"
            " * Fully self-contained: SIMDRNG_WITH_XSIMD is baked to 0, all SIMD code\n"
            " * is stripped, and no xsimd/poet/project headers are required.\n"
            " */\n\n"
        )
    return (
        "/* Auto-generated single-header for simdrng (SIMD variant).\n"
        " * Do not edit directly.\n"
        " *\n"
        " * poet is inlined; xsimd is kept as an external <xsimd/...> include, so\n"
        " * compile with the xsimd headers available (installed, or the Compiler\n"
        " * Explorer xsimd library). The *Native generators (XoshiroNative,\n"
        " * Philox4x64Native, ...) work header-only. The runtime-dispatch types\n"
        " * (XoshiroSIMD/Philox*SIMD and the default simdrng::Xoshiro alias) need the\n"
        " * compiled libsimdrng.a and are not available from this header alone.\n"
        " */\n\n"
    )


def build_single_header(input_path, output_path, root, guard, extra_roots, strip_xsimd, xsimd_default):
    search_dirs = [root / "include"] + extra_roots
    allowed_roots = [root] + extra_roots
    body = inline_file(input_path.resolve(), set(), search_dirs, allowed_roots, strip_xsimd)

    default = ""
    if xsimd_default and not strip_xsimd:
        default = "#ifndef SIMDRNG_WITH_XSIMD\n#define SIMDRNG_WITH_XSIMD 1\n#endif\n\n"

    content = (
        f"{make_banner(strip_xsimd)}"
        f"#ifndef {guard}\n#define {guard}\n\n{default}{body}\n#endif // {guard}\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"Wrote {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", "-r", default=".")
    p.add_argument("--input", "-i", default="include/simdrng/simdrng.hpp")
    p.add_argument("--output", "-o", default="single-header/simdrng.hpp")
    p.add_argument("--guard", default="SIMDRNG_SINGLE_HEADER_HPP")
    p.add_argument("--extra-root", action="append", default=[],
                   help="Additional root to inline <...> includes from (e.g. poet's include dir).")
    p.add_argument("--strip-xsimd", action="store_true",
                   help="Produce the scalar variant: evaluate SIMDRNG_WITH_XSIMD as 0 and drop SIMD code.")
    p.add_argument("--xsimd-default", action="store_true",
                   help="Default SIMDRNG_WITH_XSIMD to 1 if the consumer did not define it.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    inp = (root / args.input).resolve()
    out = (root / args.output).resolve()
    extra_roots = [Path(d).resolve() for d in args.extra_root]
    if not inp.exists():
        print(f"Input header not found: {inp}", file=sys.stderr)
        sys.exit(2)
    for d in extra_roots:
        if not d.exists():
            print(f"--extra-root not found: {d}", file=sys.stderr)
            sys.exit(2)
    build_single_header(inp, out, root, args.guard, extra_roots, args.strip_xsimd, args.xsimd_default)


if __name__ == "__main__":
    main()
