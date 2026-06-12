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
import sys
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


def is_bulk(b: dict) -> bool:
    """Bulk-fill benchmarks call ``SetItemsProcessed`` and report per-element
    throughput in ``items_per_second``. Single-value benchmarks emit one value
    per iteration and set no item count, so this field distinguishes the two."""
    return b.get("items_per_second") is not None


def latency_chart(benches: list[dict], subtitle: str, out: Path) -> None:
    """Per-call ``u64`` latency (ns) for the single-value generators.

    Each iteration is one ``operator()`` call served from the generator's
    buffered SIMD block, so ``real_time`` is the per-call latency directly.
    Lower is better. Restricted to u64 (the native output); the double paths
    (``uniform()`` vs ``std::uniform_real_distribution``) are discussed in the
    benchmarks docs.
    """
    single = [b for b in benches if not is_bulk(b) and "double" not in b["name"]]
    if not single:
        return
    names = [pretty_name(b["name"]) for b in single]
    ns = [float(b["real_time"]) for b in single]
    _bar_chart(names, ns, "Single-value u64 latency (lower is better)",
               "ns / eval", subtitle, out)


def _bulk_api(name: str) -> str:
    """Label a bulk benchmark by the buffer-fill API it calls: ``generate()``
    for u64 buffers, ``fill_uniform()`` for double buffers."""
    label = pretty_name(name)
    return label.replace("fill u64", "generate()").replace("fill double", "fill_uniform()")


def throughput_chart(benches: list[dict], subtitle: str, out: Path) -> None:
    """Buffer-fill throughput (Mop/s) for the ``generate``/``fill_uniform`` APIs.

    ``real_time`` here covers the whole buffer (thousands of elements), so the
    per-element rate lives in ``items_per_second``. Higher is better.

    Restricted to ``generate()`` on u64 buffers, Scalar vs SIMD. Scalar has no
    dedicated bulk API — it just loops ``operator()`` — but that is exactly the
    baseline SIMD ``generate()`` is compared against. Native is dropped (it
    duplicates SIMD on AVX2 hardware), and u64 is the representative workload.
    """
    bulk = [b for b in benches
            if is_bulk(b) and "Native" not in b["name"] and b["name"].endswith("_fill_u64")]
    if not bulk:
        return
    names = [_bulk_api(b["name"]) for b in bulk]
    mops = [float(b["items_per_second"]) / 1e6 for b in bulk]
    _bar_chart(names, mops, "Buffer-fill throughput (higher is better)",
               "Mop/s", subtitle, out)


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
        else:
            print(f"warning: no Scalar/SIMD pair for {base!r}; omitting from speedup chart", file=sys.stderr)
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
        latency_chart(benches, sub, args.out / "latency.svg")
        throughput_chart(benches, sub, args.out / "throughput.svg")
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
        latency_chart(benches, subtitle, out_dir / "latency.svg")
        throughput_chart(benches, subtitle, out_dir / "throughput.svg")
        scalar_vs_simd(benches, subtitle, out_dir / "scalar_vs_simd.svg")


if __name__ == "__main__":
    main()
