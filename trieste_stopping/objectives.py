from __future__ import annotations

from math import sqrt
from typing import Sequence

import tensorflow as tf
from gpflow.config import default_float
from gpflow.kernels import Kernel, Matern52
from trieste.space import Box, SearchSpace
from trieste_stopping.models import draw_kernel_feature_map, LinearTrajectory


class GaussianProcessObjective:
    def __init__(
        self,
        kernel: Kernel,
        search_space: SearchSpace,
        noise_variance: float = 0.0,
        num_features: int = 4096,
    ):
        feature_map = draw_kernel_feature_map(
            kernel=kernel,
            num_inputs=search_space.dimension,
            num_features=num_features,
        )
        self.noise_variance = noise_variance
        self.search_space = search_space
        self.trajectory = LinearTrajectory(
            feature_map=feature_map,
            weights=tf.random.normal([num_features, 1], dtype=default_float())
        )

    def objective(self, x: tf.Tensor, noisy: bool = True) -> tf.Tensor:
        values = self.trajectory(x)
        if noisy:
            rvs = tf.random.normal(shape=values.shape, dtype=values.dtype)
            values += sqrt(self.noise_variance) * rvs

        return values

    @property
    def dim(self) -> int:
        """The input dimensionality of the test function"""
        return self.search_space.dimension

    @property
    def bounds(self) -> list[list[float]]:
        """The input space bounds of the test function"""
        return [self.search_space.lower, self.search_space.upper]


class Matern52Objective(GaussianProcessObjective):
    """Convenience class for a GaussianProcessObjective with a Matern 5/2 kernel."""
    def __init__(
        self,
        dim: int,
        lengthscales: float | Sequence[float],
        noise_variance: float = 0.0,
        num_features: int = 4096,
    ):
        super().__init__(
            kernel=Matern52(lengthscales=lengthscales),
            noise_variance=noise_variance,
            search_space=Box(lower=dim * [0], upper=dim * [1]),
            num_features=num_features,
        )
