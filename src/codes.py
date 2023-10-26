from scipy.linalg import hadamard
import numpy as np
import torch
import typing
import abc


class VectorsMaker(abc.ABC):
    @abc.abstractmethod
    def __call__(self, n_vectors: int) -> np.ndarray:
        """Provides `n_vectors` vectors, stacked as a single array"""


class IndependentBinaryVectorsMaker(VectorsMaker):
    def __init__(self):
        """`VectorsMaker` for linearly independent binary vectors

        If we were to replace all `0` values by `-1`, they would be orthogonal.
        """
        self._code_width_lookup = {}

    def _get_code_width(self, n_vectors: int) -> int:
        if n_vectors not in self._code_width_lookup:
            k = np.log(n_vectors) / np.log(2)
            k_min, k_max = np.floor(k), np.ceil(k)
            if k_min == k_max:
                k = np.log(n_vectors + 1) / np.log(2)
                k_min, k_max = np.floor(k), np.ceil(k)

            if 2 ** k_min > n_vectors:
                k = k_min
            else:
                k = k_max
            self._code_width_lookup[n_vectors] = (
                k_min if 2 ** k_min > n_vectors else k_max
            )
        return self._code_width_lookup[n_vectors]

    def __call__(self, n_vectors: int) -> np.ndarray:
        code_width = self._get_code_width(n_vectors)
        out = hadamard(n=2 ** code_width)[1 : n_vectors + 1]
        rng = np.random.default_rng()
        rng.shuffle(out, axis=0)
        rng.shuffle(out, axis=1)
        return out


default_code_maker = IndependentBinaryVectorsMaker()

class UniqueBinaryVectorsMaker(VectorsMaker):
    def __init__(
        self, code_width: int, *, max_retry: int = 20, free_density: float = 0.5
    ):
        """`VectorsMaker` for unique binary vectors

        Many factors could affect which `code_width` to use, so no default is provided.

        If the randomly-generated vectors are not unique, up to `max_retry` attempts
        re made to find unique replacements.

        The `free_density` is the expected density (probability that an entry is nonzero)
        when no "retry" occur.

        The probability that no "retry" occurs is given by `birthday_collision_probability(2**code_width, n_vectors)`.
        """
        self._code_width = code_width
        self._max_retry = max_retry
        self._free_density = free_density

    def __call__(self, n_vectors: int) -> np.ndarray:
        out = set()
        attempt = 0
        rng = np.random.default_rng()
        while len(out) < n_vectors:
            if attempt > self._max_retry:
                raise RuntimeError(f"Reached max_retry={self._max_retry} attempts")
            out.update(
                tuple(v)
                for v in (
                    rng.random(size=(n_vectors - len(out), self._code_width))
                    < self._free_density
                )
            )
            attempt += 1
        out = np.array(list(out))
        rng.shuffle(out)
        assert out.shape == (n_vectors, self._code_width)
        return out


def get_code(
    n_classes: int, n_classifiers: int, code_maker: typing.Optional[VectorsMaker] = None
) -> torch.Tensor:
    """Provide code for all classifiers

    Leaving `code_maker` to its default should be equivalent to calling `utils.get_codes_matrix`.

    >>> get_code(1000, 4).shape
    torch.Size([1000, 4096])
    >>> get_code(1000, 4, UniqueBinaryVectorsMaker(16)).shape
    torch.Size([1000, 64])
    """
    if code_maker is None:
        code_maker = default_code_maker
    assert n_classes > 1
    assert n_classifiers > 0
    codes = np.concatenate(
        list(code_maker(n_classes) for _ in range(n_classifiers)), axis=1
    )
    codes = torch.Tensor(codes).float()
    codes[codes == 0] = -1.0
    return codes


if __name__ == "__main__":
    import doctest

    doctest.testmod()
