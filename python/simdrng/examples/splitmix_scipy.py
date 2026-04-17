"""SplitMix with SciPy distributions."""

from scipy import stats
import simdrng


def main() -> None:
    rng = simdrng.SplitMix(123)
    print("normals:", stats.norm.rvs(size=5, random_state=rng))
    print("integers:", stats.randint.rvs(0, 10, size=5, random_state=rng))


if __name__ == "__main__":
    main()
