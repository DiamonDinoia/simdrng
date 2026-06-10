#!/usr/bin/env python3
"""Audit vectorisation and register spills in compiled simdrng code.

Disassembles a binary or object file, isolates the hot generator functions, and
for each reports the widest vector register used (xmm/ymm/zmm) plus a count of
stack spills (vector loads/stores against %rsp/%rbp). It flags:

  * de-vectorisation -- a *SIMD* function that never touches ymm/zmm, i.e. the
    dispatch tier silently dropped to scalar/SSE width;
  * spills -- vector <-> stack traffic, a proxy for register pressure (the
    Philox4x64 SIMD wide path is the known offender).

Adapted from POET's scripts/extract_asm.py (objdump selection + function
splitting); the vectorisation/spill report is simdrng-specific.

Usage:
    python3 tools/asm_audit.py BINARY [--compiler gcc-15] [--md report.md]
                                      [--json report.json] [--filter REGEX]
Exit status is non-zero only if --fail-on-devectorisation is set and a SIMD
function devectorised; spills are always reported, never fatal (they are a
tuning signal, not a correctness bug).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Hot generator entry points worth auditing. Demangled names contain these.
DEFAULT_FILTER = r"(XoshiroSIMD|ChaChaSIMD|PhiloxSIMD|Native|gen_block|populate_cache|next_block|store_blocks)"

VECTOR_REG = re.compile(r"%(xmm|ymm|zmm)\d+")
# A vector move whose memory operand is stack-relative => spill/reload.
SPILL = re.compile(r"\b(v?mov(?:aps|ups|dqa|dqu|apd|upd))\b.*[-0-9]*\(%r(?:sp|bp)\)")
FUNC_LABEL = re.compile(r"^[0-9a-f]+ <(.+)>:\s*$")


def find_objdump(compiler: str) -> str:
    if compiler.startswith("clang"):
        ver = compiler.partition("-")[2]
        for cand in (f"llvm-objdump-{ver}", "llvm-objdump"):
            if shutil.which(cand):
                return cand
    return "objdump"


def disassemble(objdump: str, binary: str) -> str:
    cmd = [objdump, "-d", "--demangle", binary]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout


def split_functions(disasm: str) -> list[tuple[str, list[str]]]:
    funcs: list[tuple[str, list[str]]] = []
    name, body = None, []
    for line in disasm.splitlines():
        m = FUNC_LABEL.match(line)
        if m:
            if name is not None:
                funcs.append((name, body))
            name, body = m.group(1), []
        elif name is not None:
            body.append(line)
    if name is not None:
        funcs.append((name, body))
    return funcs


def audit_function(name: str, body: list[str]) -> dict:
    widths = {"xmm": 0, "ymm": 0, "zmm": 0}
    spills = 0
    for line in body:
        for m in VECTOR_REG.finditer(line):
            widths[m.group(1)] += 1
        if SPILL.search(line):
            spills += 1
    width = "zmm" if widths["zmm"] else "ymm" if widths["ymm"] else "xmm" if widths["xmm"] else "scalar"
    # Only the hot block-fill path should be vector-wide. The cold one-shot
    # setup code (the dispatch InitFunctor lambdas / _FUN trampolines) is scalar
    # by nature, so it is not a de-vectorisation regression.
    is_cold_setup = any(tok in name for tok in ("Init", "_FUN"))
    is_simd = "SIMD" in name and not is_cold_setup
    devectorised = is_simd and width in ("scalar", "xmm")
    return {
        "function": name,
        "width": width,
        "xmm": widths["xmm"],
        "ymm": widths["ymm"],
        "zmm": widths["zmm"],
        "spills": spills,
        "devectorised": devectorised,
    }


def to_markdown(rows: list[dict]) -> str:
    out = ["| function | width | spills | flags |", "|---|:--:|--:|---|"]
    for r in rows:
        flags = []
        if r["devectorised"]:
            flags.append("⚠ de-vectorised")
        if r["spills"]:
            flags.append(f"{r['spills']} spill(s)")
        short = r["function"]
        if len(short) > 70:
            short = short[:67] + "..."
        out.append(f"| `{short}` | {r['width']} | {r['spills']} | {', '.join(flags) or '—'} |")
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("binary")
    p.add_argument("--compiler", default="gcc")
    p.add_argument("--filter", default=DEFAULT_FILTER)
    p.add_argument("--md", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--fail-on-devectorisation", action="store_true")
    args = p.parse_args()

    if not Path(args.binary).exists():
        print(f"error: binary not found: {args.binary}", file=sys.stderr)
        return 2

    disasm = disassemble(find_objdump(args.compiler), args.binary)
    if not disasm:
        print(f"error: empty disassembly for {args.binary}", file=sys.stderr)
        return 2

    pat = re.compile(args.filter)
    rows = [
        audit_function(name, body)
        for name, body in split_functions(disasm)
        if pat.search(name)
    ]
    rows.sort(key=lambda r: (-r["spills"], r["function"]))

    md = to_markdown(rows)
    if args.md:
        args.md.write_text(md)
    else:
        print(md)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))

    devec = [r["function"] for r in rows if r["devectorised"]]
    if devec:
        print(f"\nde-vectorised SIMD functions: {len(devec)}", file=sys.stderr)
        for f in devec:
            print(f"  {f}", file=sys.stderr)
    total_spills = sum(r["spills"] for r in rows)
    print(f"audited {len(rows)} functions; {total_spills} total vector stack spills", file=sys.stderr)

    return 1 if (args.fail_on_devectorisation and devec) else 0


if __name__ == "__main__":
    raise SystemExit(main())
