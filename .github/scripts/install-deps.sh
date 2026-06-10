#!/usr/bin/env bash
# Install the exact xsimd and poet sources that a simdrng build already fetched
# (via CPM) into a prefix, so a downstream find_package(simdrng) can resolve its
# find_dependency(xsimd)/find_dependency(poet) calls. Reading the source dirs
# from the build's CMakeCache keeps the dependency pins single-sourced in
# CMakeLists.txt rather than duplicated here.
#
# Usage: install-deps.sh <simdrng-build-dir> <install-prefix>
set -euo pipefail

build_dir=${1:?usage: install-deps.sh <build-dir> <prefix>}
prefix=${2:?usage: install-deps.sh <build-dir> <prefix>}
cache="$build_dir/CMakeCache.txt"

xsimd_src=$(grep -m1 'xsimd_SOURCE_DIR' "$cache" | cut -d= -f2)
poet_src=$(grep -m1 'CPM_PACKAGE_poet_SOURCE_DIR' "$cache" | cut -d= -f2)
echo "xsimd source: $xsimd_src"
echo "poet  source: $poet_src"

cmake -S "$xsimd_src" -B "$build_dir/_dep-xsimd" -DCMAKE_INSTALL_PREFIX="$prefix" \
	-DBUILD_TESTS=OFF -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=OFF
cmake --install "$build_dir/_dep-xsimd"

cmake -S "$poet_src" -B "$build_dir/_dep-poet" -DCMAKE_INSTALL_PREFIX="$prefix" \
	-DBUILD_TESTING=OFF -DPOET_BUILD_TESTS=OFF \
	-DPOET_BUILD_BENCHMARKS=OFF -DPOET_BUILD_EXAMPLES=OFF
cmake --install "$build_dir/_dep-poet"
