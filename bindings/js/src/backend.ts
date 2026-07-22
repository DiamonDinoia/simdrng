// backend.ts — the backend-agnostic contract plus runtime backend selection.
//
// simdrng's JS binding has two interchangeable backends over the same C ABI:
//   * "native": a Node-API `.node` addon (fast, Node-only) — simdrng_napi.cpp
//   * "wasm":   an Emscripten module (browser + Node) — wasm.ts / wasm_glue.cpp
// Both implement the `Backend` interface below; `index.ts` (the Generator class)
// is written entirely against it and never imports a backend directly.

/** Generator kinds, ordered to match simdrng_kind in capi.h (index == C enum). */
export const KINDS = [
  "splitmix",
  "xoshiro",
  "chacha8",
  "chacha12",
  "chacha20",
  "philox4x32",
  "philox2x32",
  "philox4x64",
  "philox2x64",
] as const;
export type Kind = (typeof KINDS)[number];

/** A live generator handle. Draws run the generator's own SIMD path. */
export interface BackendGenerator {
  /** 0-based C enum the handle was created with. */
  readonly kind: number;
  /** `n` uniform Float64 draws in [0,1). */
  random(n: number): Float64Array;
  /** `n` raw 64-bit words. */
  raw(n: number): BigUint64Array;
  free(): void;
}

export interface Backend {
  readonly name: "native" | "wasm";
  readonly versionString: string;
  create(kind: number, seed: bigint): BackendGenerator;
}

export type BackendChoice = "auto" | "native" | "wasm";

/**
 * Load a backend. `"auto"` (the default) prefers the native addon under Node and
 * falls back to WASM if it is unavailable; in non-Node environments it always
 * uses WASM. `"native"` / `"wasm"` force a specific backend (the test suite
 * exercises both under Node).
 */
export async function loadBackend(choice: BackendChoice = "auto"): Promise<Backend> {
  const isNode =
    typeof process !== "undefined" &&
    !!(process as { versions?: { node?: string } }).versions?.node;

  if (choice === "native" || (choice === "auto" && isNode)) {
    try {
      const { makeNativeBackend } = await import("./native.js");
      return makeNativeBackend();
    } catch (err) {
      if (choice === "native") throw err;
      // fall through to WASM
    }
  }

  const { makeWasmBackend } = await import("./wasm.js");
  return makeWasmBackend();
}
