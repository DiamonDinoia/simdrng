"""XoshiroSIMD with thread/cluster offsets for parallel simulations."""

import numpy as np
import pyrandom


def main() -> None:
    # Thread-offset seeding: same seed, different streams
    streams = [pyrandom.XoshiroSIMD(seed=42, thread=i) for i in range(4)]

    for i, rng in enumerate(streams):
        vals = rng.random(5)
        print(f"thread {i}: {vals}")

    # Cluster-offset for distributed computing
    rng = pyrandom.XoshiroSIMD(seed=42, thread=0, cluster=1)
    print(f"cluster 1: {rng.random(5)}")


if __name__ == "__main__":
    main()
