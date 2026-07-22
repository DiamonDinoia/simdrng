// index.ts — the public simdrng JS/TS API.
//
// `Generator.create(kind, seed, opts?)` resolves to a generator handle; draw
// uniform reals with `.random(n)` or raw 64-bit words with `.raw(n)`. The
// backend (native `.node` addon or WASM) is chosen at runtime by backend.ts and
// is invisible here.

import {
  type Backend,
  type BackendChoice,
  type BackendGenerator,
  KINDS,
  type Kind,
  loadBackend,
} from "./backend.js";

export { KINDS, loadBackend };
export type { BackendChoice, Kind };

// One backend instance per choice, reused across generators (WASM instantiation
// is not free; the native addon is a singleton anyway).
const backendCache = new Map<BackendChoice, Promise<Backend>>();
function getBackend(choice: BackendChoice): Promise<Backend> {
  let cached = backendCache.get(choice);
  if (!cached) {
    cached = loadBackend(choice);
    backendCache.set(choice, cached);
  }
  return cached;
}

export interface GeneratorOptions {
  /** Force a backend; defaults to "auto" (native under Node, else WASM). */
  backend?: BackendChoice;
}

export class Generator {
  private constructor(
    private readonly gen: BackendGenerator,
    readonly kind: Kind,
    readonly libVersion: string,
  ) {}

  /** Create a generator of `kind` seeded with the 64-bit `seed` (BigInt or number). */
  static async create(
    kind: Kind,
    seed: bigint | number = 0n,
    opts: GeneratorOptions = {},
  ): Promise<Generator> {
    const code = KINDS.indexOf(kind);
    if (code < 0) throw new Error(`unknown generator kind "${kind}"; expected one of ${KINDS}`);
    const backend = await getBackend(opts.backend ?? "auto");
    const gen = backend.create(code, BigInt.asUintN(64, BigInt(seed)));
    return new Generator(gen, kind, backend.versionString);
  }

  /** `n` uniform Float64 draws in [0,1). */
  random(n: number): Float64Array {
    return this.gen.random(n);
  }

  /** `n` raw 64-bit words straight from the generator. */
  raw(n: number): BigUint64Array {
    return this.gen.raw(n);
  }

  free(): void {
    this.gen.free();
  }
  [Symbol.dispose](): void {
    this.free();
  }
}

export default Generator;
