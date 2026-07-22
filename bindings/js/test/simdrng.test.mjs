// simdrng.test.mjs — node:test suite, run by CTest as `js_simdrng`.
//
// Exercises BOTH backends under Node: the native .node addon (built by the
// bindings-js preset) and the WASM module (built by bindings-js-wasm). The WASM
// block is skipped when its artifacts are absent (native-only local build
// without emsdk). Coverage mirrors the Fortran/Julia tests: identity,
// determinism, stream divergence, and the [0,1) range contract, for all kinds.

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, test } from "node:test";

import { Generator, KINDS } from "../dist/index.js";

const SEED = 0x9e3779b97f4a7c15n;
const N = 1024;

const hasNative = existsSync(fileURLToPath(new URL("../dist/simdrng.node", import.meta.url)));
const hasWasm = existsSync(fileURLToPath(new URL("../dist/simdrng.mjs", import.meta.url)));
const backends = [...(hasNative ? ["native"] : []), ...(hasWasm ? ["wasm"] : [])];

test("at least one backend was built", () => {
  assert.ok(
    backends.length > 0,
    "no backend artifacts found in dist/ (build bindings-js and/or bindings-js-wasm)",
  );
});

const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

for (const backend of backends) {
  describe(`simdrng [${backend}]`, () => {
    for (const kind of KINDS) {
      test(kind, async () => {
        const g = await Generator.create(kind, SEED, { backend });
        assert.equal(g.kind, kind);
        assert.ok(g.libVersion.length > 0);

        // determinism: same seed -> identical raw stream
        const a = g.raw(N);
        const g2 = await Generator.create(kind, SEED, { backend });
        assert.ok(eq(a, g2.raw(N)), "raw stream not deterministic");

        // divergence: different seed -> different stream
        const g3 = await Generator.create(kind, SEED + 1n, { backend });
        assert.ok(!eq(a, g3.raw(N)), "distinct seeds gave identical stream");

        // range contract for the uniform reals
        const d = (await Generator.create(kind, SEED, { backend })).random(N);
        assert.equal(d.length, N);
        assert.ok(
          d.every((x) => x >= 0 && x < 1),
          "random out of [0,1)",
        );

        assert.ok(a instanceof BigUint64Array);
        for (const h of [g, g2, g3]) h.free();
      });
    }
  });
}
