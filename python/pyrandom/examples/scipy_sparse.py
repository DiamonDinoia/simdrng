"""SciPy sparse matrix generation with pyrandom."""

from scipy import sparse
import pyrandom


def main() -> None:
    rng = pyrandom.XoshiroSIMD(2024)
    mat = sparse.random(5, 5, density=0.4, random_state=rng)
    print("sparse matrix:\n", mat.toarray().round(3))


if __name__ == "__main__":
    main()
