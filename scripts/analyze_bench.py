#!/usr/bin/env python3
"""Turn a Google Benchmark JSON file into a markdown + CSV summary.

Reads --benchmark_format=json output and emits a table with the columns
we care about for simdrng: ns/op, op/s, IPC (when libpfm is
available), cache-miss rate, and branch-miss rate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def load(path: Path) -> list[dict]:
    with path.open() as f:
        doc = json.load(f)
    return doc.get("benchmarks", [])


def rows(benches: list[dict]) -> list[dict]:
    out = []
    for b in benches:
        if b.get("run_type") != "iteration":
            continue
        ns = float(b.get("real_time", 0.0))
        cycles = float(b.get("CYCLES", 0) or 0)
        instr = float(b.get("INSTRUCTIONS", 0) or 0)
        cache_miss = float(b.get("CACHE-MISSES", 0) or 0)
        branch = float(b.get("BRANCHES", 0) or 0)
        branch_miss = float(b.get("BRANCH-MISSES", 0) or 0)
        out.append({
            "name": b["name"],
            "ns_per_op": ns,
            "op_per_s": _safe_div(1e9, ns),
            "ipc": _safe_div(instr, cycles),
            "cache_miss_pct": 100.0 * _safe_div(cache_miss, instr),
            "branch_miss_pct": 100.0 * _safe_div(branch_miss, branch),
        })
    return out


def to_markdown(rs: list[dict]) -> str:
    hdr = "| benchmark | ns/op | Mop/s | IPC | cache-miss% | branch-miss% |\n"
    sep = "|---|---:|---:|---:|---:|---:|\n"
    body = "".join(
        f"| {r['name']} | {r['ns_per_op']:.2f} | {r['op_per_s']/1e6:.1f} | "
        f"{r['ipc']:.2f} | {r['cache_miss_pct']:.2f} | {r['branch_miss_pct']:.2f} |\n"
        for r in rs
    )
    return hdr + sep + body


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("json_file", type=Path)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--md", type=Path, default=None)
    args = p.parse_args()

    benches = load(args.json_file)
    rs = rows(benches)

    md = to_markdown(rs)
    if args.md:
        args.md.write_text(md)
    else:
        print(md)

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()) if rs else [])
            w.writeheader()
            w.writerows(rs)


if __name__ == "__main__":
    main()
