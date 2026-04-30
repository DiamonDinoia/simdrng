#!/usr/bin/env python3
"""Generate SVG charts from Google Benchmark JSON output.

Usage:
    python scripts/generate_charts.py bench.json [--out charts/]
    python scripts/generate_charts.py results/    # walks compiler dirs

If a ``cpu_model.txt`` sits next to the JSON (or in the compiler dir), its
contents are shown as a subtitle on every chart.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


VARIANTS = ("Scalar", "SIMD", "Native")


def load_one(path: Path) -> list[dict]:
    with path.open() as f:
        doc = json.load(f)
    return [b for b in doc.get("benchmarks", []) if b.get("run_type") == "iteration"]


def cpu_label(json_path: Path) -> str:
    """Look for cpu_model.txt next to the JSON or in its parent dir."""
    for candidate in (json_path.with_name("cpu_model.txt"),
                      json_path.parent / "cpu_model.txt",
                      json_path.parent.parent / "cpu_model.txt"):
        if candidate.is_file():
            text = candidate.read_text().strip()
            if text:
                return text
    # Fall back to JSON context if present.
    try:
        with json_path.open() as f:
            ctx = json.load(f).get("context", {})
        cpu = ctx.get("cpu_info") or {}
        brand = cpu.get("brand") or cpu.get("model") or ""
        if brand:
            return brand
        n = ctx.get("num_cpus")
        mhz = ctx.get("mhz_per_cpu")
        if n and mhz:
            return f"{n} CPU(s) @ {mhz} MHz"
    except (OSError, ValueError):
        pass
    return ""


def pretty_name(raw: str) -> str:
    """Turn ``BM_XoshiroSIMD_std_double`` into ``Xoshiro · SIMD · std double``."""
    n = raw[3:] if raw.startswith("BM_") else raw
    variant = ""
    for v in VARIANTS:
        m = re.search(rf"(?<=[A-Za-z0-9]){v}(?=_|$)", n)
        if m:
            variant = v
            n = n[: m.start()] + n[m.end():]
            break
    n = n.strip("_")
    parts = n.split("_", 1)
    gen = parts[0]
    suffix = parts[1].replace("_", " ") if len(parts) > 1 else ""
    out = [gen]
    if variant:
        out.append(variant)
    if suffix:
        out.append(suffix)
    return " · ".join(out)


def _bar_chart(names: list[str], values: list[float], title: str, ylabel: str,
               subtitle: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5 + 0.15 * len(names)))
    ax.barh(names, values, color="#3178c6")
    ax.set_xlabel(ylabel)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    else:
        ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def overview_chart(benches: list[dict], subtitle: str, out: Path) -> None:
    names = [pretty_name(b["name"]) for b in benches]
    op_per_s = [1e9 / float(b["real_time"]) / 1e6 for b in benches]  # Mop/s
    _bar_chart(names, op_per_s, "Throughput (Mop/s)", "Mop/s", subtitle, out)


def scalar_vs_simd(benches: list[dict], subtitle: str, out: Path) -> None:
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
            labels.append(pretty_name(base))
            speedups.append(d["SIMD"] / d["Scalar"])
    if not labels:
        return
    _bar_chart(labels, speedups, "SIMD speedup vs scalar", "speedup x",
               subtitle, out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="JSON file or results/ dir")
    p.add_argument("--out", type=Path, default=Path("charts"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.path.is_file():
        benches = load_one(args.path)
        sub = cpu_label(args.path)
        overview_chart(benches, sub, args.out / "overview.svg")
        scalar_vs_simd(benches, sub, args.out / "scalar_vs_simd.svg")
        return

    for compiler_dir in sorted(p for p in args.path.iterdir() if p.is_dir()):
        jsons = sorted(compiler_dir.glob("*.json"))
        if not jsons:
            continue
        latest = jsons[-1]
        benches = load_one(latest)
        cpu = cpu_label(latest)
        subtitle = f"{compiler_dir.name} · {cpu}" if cpu else compiler_dir.name
        out_dir = args.out / compiler_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        overview_chart(benches, subtitle, out_dir / "overview.svg")
        scalar_vs_simd(benches, subtitle, out_dir / "scalar_vs_simd.svg")


if __name__ == "__main__":
    main()
