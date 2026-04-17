#!/usr/bin/env bash
# Build the benchmark harness against each available compiler and dump JSON
# results into results/<compiler>/<timestamp>.json. Intended for local
# reproduction of the CI benchmark matrix.
COMPILERS=("${COMPILERS:-gcc-14 g++-14 clang++-21}")
RESULTS_DIR="${RESULTS_DIR:-results}"
BUILD_DIR="${BUILD_DIR:-build-bench}"
PARALLEL="${PARALLEL:-8}"
PERF_COUNTERS="${PERF_COUNTERS:-CYCLES,INSTRUCTIONS,CACHE-MISSES,BRANCH-MISSES,BRANCHES}"

mkdir -p "$RESULTS_DIR"

for cxx in "${COMPILERS[@]}"; do
    if ! command -v "$cxx" >/dev/null 2>&1; then
        echo "skip $cxx (not found)"
        continue
    fi

    name="${cxx##*/}"
    dir="${BUILD_DIR}/${name}"
    out_dir="${RESULTS_DIR}/${name}"
    mkdir -p "$out_dir"

    cmake -S . -B "$dir" \
        -DCMAKE_CXX_COMPILER="$cxx" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_TESTS=ON -DENABLE_PYTHON=OFF -DMARCH_NATIVE=ON
    cmake --build "$dir" --target benchmarks -j "$PARALLEL"

    ts=$(date -u +%Y%m%dT%H%M%SZ)
    "$dir/tests/benchmarks" \
        --benchmark_perf_counters="$PERF_COUNTERS" \
        --benchmark_format=json \
        --benchmark_out="$out_dir/${ts}.json"
done
