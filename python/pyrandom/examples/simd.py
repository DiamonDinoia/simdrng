"""XoshiroSIMD bulk fill — transparent C++ fast path."""

import numpy as np
import pyrandom


def main() -> None:
    rng = pyrandom.XoshiroSIMD(42)

    # These all go through the C++ bulk fill automatically:
    out64 = rng.random(1_000_000)                          # float64
    out32 = rng.random(1_000_000, dtype=np.float32)        # float32 (2 per uint64)

    print(f"float64: {out64.shape}, range [{out64.min():.4f}, {out64.max():.4f})")
    print(f"float32: {out32.shape}, range [{out32.min():.4f}, {out32.max():.4f})")

    # Pre-allocated output — zero copy, GIL released
    buf = np.empty((1000, 1000), dtype=np.float64)
    rng.random(buf.shape, out=buf)
    print(f"pre-allocated: {buf.shape}, mean={buf.mean():.4f}")


if __name__ == "__main__":
    main()
