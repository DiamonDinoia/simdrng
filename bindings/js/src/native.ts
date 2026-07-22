// native.ts — the "native" backend: a thin wrapper over the Node-API addon
// (simdrng.node, built from simdrng_napi.cpp). The addon's create() already
// returns objects shaped like BackendGenerator, so this only locates the addon.

import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import type { Backend, BackendGenerator } from "./backend.js";

interface Addon {
  create(kind: number, seed: bigint): BackendGenerator;
  versionString: string;
}

// Resolve the N-API addon. Published packages carry per-platform prebuilt
// binaries under prebuilds/<platform>-<arch>/ (node-gyp-build picks the match;
// N-API is ABI-stable, so one binary per platform serves every Node version). A
// local CMake build instead drops simdrng.node next to this file in dist/.
function loadAddon(require: NodeRequire): Addon {
  const here = dirname(fileURLToPath(import.meta.url));
  try {
    const gypBuild = require("node-gyp-build") as (dir: string) => Addon;
    return gypBuild(join(here, "..")); // package root holds prebuilds/
  } catch {
    return require("./simdrng.node") as Addon;
  }
}

export function makeNativeBackend(): Backend {
  const require = createRequire(import.meta.url);
  const addon = loadAddon(require);
  return {
    name: "native",
    versionString: addon.versionString,
    create: (kind: number, seed: bigint) => addon.create(kind, seed),
  };
}
