from __future__ import annotations

from itertools import zip_longest
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
import tensorflow as tf
from gpflow.base import TensorType
from gpflow.config import default_float
from tensorflow_probability.python.bijectors import Bijector, Chain, Shift, Scale
from trieste.space import Box, Constraint, SearchSpace

SpaceType = TypeVar("SpaceType")


class UnitHypercube(Box):
    """Unit hypercube space with an affine map from [0, 1] to [lower, upper]."""
    bijector: Bijector

    def __init__(
        self,
        lower: Sequence[float] | TensorType,
        upper: Sequence[float] | TensorType,
        constraints: Sequence[Constraint] | None = None,
        ctol: float | TensorType = 1e-7,
    ):
        super().__init__(
            lower=[0 for _ in lower],
            upper=[1 for _ in upper],
            constraints=constraints,
            ctol=ctol,
        )
        lower = tf.convert_to_tensor(lower, dtype_hint=default_float())
        upper = tf.convert_to_tensor(upper, dtype_hint=default_float())
        self.bijector = Chain(bijectors=(Shift(lower), Scale(upper - lower)))

    def from_unit(self, x: TensorType) -> TensorType:
        return self.bijector(x)

    def to_unit(self, x: TensorType) -> TensorType:
        return self.bijector.inverse(x)


class NamedSpaceWrapper(Generic[SpaceType]):
    """SearchSpace wrapper with methods for converting to and from dictionaries."""
    def __init__(self, names: tuple[str, ...], space: SpaceType):
        self.names = names
        self.space = space

    def __getattr__(self, name: str, default: Any | None = None):
        return getattr(self.space, name, default)

    def to_dict(self, x: TensorType) -> dict[str, np.ndarray]:
        return dict(zip_longest(self.names, tf.unstack(x, axis=-1)))

    def from_dict(self, x: dict[str, float]) -> tf.Tensor:
        return tf.convert_to_tensor(
            (x[name] for name in self.names), dtype_hint=default_float()
        )
