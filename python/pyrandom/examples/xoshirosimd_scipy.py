"""XoshiroSIMD with SciPy distributions."""

from scipy import stats
import pyrandom


def main() -> None:
    rng = pyrandom.XoshiroSIMD(123)
    print("normals:", stats.norm.rvs(size=5, random_state=rng))
    print("integers:", stats.randint.rvs(0, 10, size=5, random_state=rng))


if __name__ == "__main__":
    main()
