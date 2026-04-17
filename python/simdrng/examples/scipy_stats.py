"""SciPy integration — any simdrng generator works as random_state."""

from scipy import stats
import simdrng


def main() -> None:
    rng = simdrng.XoshiroSIMD(2024)

    # All scipy.stats distributions accept our generators
    normals = stats.norm.rvs(size=5, random_state=rng)
    print("normal:", normals)

    poisson = stats.poisson.rvs(mu=3.0, size=5, random_state=rng)
    print("poisson:", poisson)

    # ChaCha for cryptographic-quality randomness
    rng_chacha = simdrng.ChaCha20(seed=2024)
    uniform = stats.uniform.rvs(size=5, random_state=rng_chacha)
    print("ChaCha20 uniform:", uniform)


if __name__ == "__main__":
    main()
