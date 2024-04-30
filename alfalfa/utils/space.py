import numpy as np
from beartype.cave import IntType
from beartype.typing import Generic, Optional, Sequence, TypeVar
from jaxtyping import Int, Shaped

BoundType = int | float | str
B = TypeVar("B", int, float, str)


class Dimension(Generic[B]):
    var_type = ""
    is_bin = False

    def __init__(self, bnds: list[B], key: str):
        self.bnds = bnds
        self.key = key

    def transform(self, x: Shaped[np.ndarray, "N"]) -> Shaped[np.ndarray, "N"]:
        return x

    def sample(self, n: int, rng: np.random.Generator) -> Shaped[np.ndarray, "N"]:
        pass

    def grid(self, _shape: IntType):
        return np.array(self.bnds)

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.key!r}, bounds={self.bnds!r})"


class ContinuousDimension(Dimension[float]):
    var_type = "conti"

    def sample(self, n: int, rng: np.random.Generator) -> Shaped[np.ndarray, "N"]:
        return rng.uniform(*self.bnds, size=n)

    def grid(self, shape: IntType):
        return np.linspace(*self.bnds, shape)


class IntegerDimension(Dimension[int]):
    var_type = "int"

    @property
    def is_bin(self):
        return self.bnds == [0, 1]

    def sample(self, n: int, rng: np.random.Generator) -> Shaped[np.ndarray, "N"]:
        return rng.integers(*self.bnds, size=n, endpoint=True)


class CategoricalDimension(Dimension[BoundType]):
    var_type = "cat"

    def transform(self, x: Shaped[np.ndarray, "N"]) -> Int[np.ndarray, "N"]:
        return np.array([self.bnds.index(xi) for xi in x])

    def sample(self, n: int, rng: np.random.Generator) -> Shaped[np.ndarray, "N"]:
        return rng.choice(self.bnds, size=n, replace=True)


class Space:
    def __init__(self, dims: list[Dimension]):
        self.cat_idx = [
            idx for idx, dim in enumerate(dims) if isinstance(dim, CategoricalDimension)
        ]
        self.int_idx = [
            idx for idx, dim in enumerate(dims) if isinstance(dim, IntegerDimension)
        ]
        self.cont_idx = [
            idx for idx in range(len(dims)) if idx not in self.cat_idx + self.int_idx
        ]
        self.dims = dims

        self._key_to_idx = {dim.key: idx for idx, dim in enumerate(self.dims)}

    @classmethod
    def from_bounds(
        cls,
        bnds: list[list[BoundType]],
        cat_idx: Optional[list[int]] = None,
        int_idx: Optional[list[int]] = None,
    ):
        cat_idx = cat_idx or []
        int_idx = int_idx or []

        dims = []
        for idx, b in enumerate(bnds):
            key = f"x{idx}"
            if idx in cat_idx:
                dims.append(CategoricalDimension(b, key))
            elif idx in int_idx:
                dims.append(IntegerDimension(b, key))
            else:
                dims.append(ContinuousDimension(b, key))

        return cls(dims)

    @property
    def bounds(self):
        return [d.bnds for d in self.dims]

    @property
    def keys(self):
        return [d.key for d in self.dims]

    def process_vals(self, X):
        pass

    def key_to_idx(self, keys: str | list[str]) -> int | list[int]:
        if isinstance(keys, str):
            return self._key_to_idx[keys]
        else:
            return [self._key_to_idx[k] for k in keys]

    def transform(self, x: Shaped[np.ndarray, "N D"]) -> Shaped[np.ndarray, "N D"]:
        x_transform = np.zeros_like(x)
        for i, dim in enumerate(self.dims):
            x_transform[:, i] = dim.transform(x[:, i])
        return x_transform.astype(float)

    def sample(self, n: int, rng: np.random.Generator) -> Shaped[np.ndarray, "N D"]:
        return np.stack([dim.sample(n, rng) for dim in self.dims], axis=-1)

    def grid(
        self, shape: Sequence[IntType] | IntType
    ) -> tuple[Shaped[np.ndarray, "N D"]]:
        """Return data sampled on a grid"""
        if isinstance(shape, IntType):
            shape = [shape] * len(self.dims)

        assert len(shape) == len(self.dims), "Shape must match number of dimensions"

        xs = [dim.grid(s) for s, dim in zip(shape, self.dims)]
        test_x_mgrid = np.meshgrid(*xs, indexing="ij")
        flats = [x.flatten() for x in test_x_mgrid]
        test_x = np.stack(flats, axis=-1)

        return test_x

    def __getitem__(self, keys: str | list[str]):
        return self.key_to_idx(keys)

    def __repr__(self):
        return repr(self.dims)

    def __len__(self):
        return len(self.dims)
