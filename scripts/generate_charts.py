#!/usr/bin/env python3
"""Generate SVG charts from Google Benchmark JSON output.

Usage:
    python scripts/generate_charts.py bench.json [--out charts/]
    python scripts/generate_charts.py results/    # walks compiler dirs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_one(path: Path) -> list[dict]:
    with path.open() as f:
        doc = json.load(f)
    return [b for b in doc.get("benchmarks", []) if b.get("run_type") == "iteration"]


def _bar_chart(names: list[str], values: list[float], title: str, ylabel: str,
               out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5 + 0.15 * len(names)))
    ax.barh(names, values, color="#3178c6")
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def overview_chart(benches: list[dict], out: Path) -> None:
    names = [b["name"] for b in benches]
    op_per_s = [1e9 / float(b["real_time"]) / 1e6 for b in benches]  # Mop/s
    _bar_chart(names, op_per_s, "Throughput (Mop/s)", "Mop/s", out)


def scalar_vs_simd(benches: list[dict], out: Path) -> None:
    pairs: dict[str, dict[str, float]] = {}
    for b in benches:
        name = b["name"]
        throughput = 1e9 / float(b["real_time"])
        for kind in ("Scalar", "SIMD"):
            if kind in name:
                base = name.replace(kind, "").replace("__", "_").strip("_")
                pairs.setdefault(base, {})[kind] = throughput
    labels, speedups = [], []
    for base, d in pairs.items():
        if "Scalar" in d and "SIMD" in d and d["Scalar"]:
            labels.append(base)
            speedups.append(d["SIMD"] / d["Scalar"])
    if not labels:
        return
    _bar_chart(labels, speedups, "SIMD speedup vs scalar", "speedup x", out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="JSON file or results/ dir")
    p.add_argument("--out", type=Path, default=Path("charts"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.path.is_file():
        benches = load_one(args.path)
        overview_chart(benches, args.out / "overview.svg")
        scalar_vs_simd(benches, args.out / "scalar_vs_simd.svg")
        return

    # Directory: iterate <compiler>/*.json and emit per-compiler charts.
    for compiler_dir in sorted(p for p in args.path.iterdir() if p.is_dir()):
        jsons = sorted(compiler_dir.glob("*.json"))
        if not jsons:
            continue
        benches = load_one(jsons[-1])  # latest
        sub = args.out / compiler_dir.name
        sub.mkdir(parents=True, exist_ok=True)
        overview_chart(benches, sub / "overview.svg")
        scalar_vs_simd(benches, sub / "scalar_vs_simd.svg")


if __name__ == "__main__":
    main()
