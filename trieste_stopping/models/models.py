from __future__ import annotations

from typing import Any

from gpflow.models import GPR
from tensorflow_probability.python import bijectors
from trieste.data import Dataset
from trieste.types import TensorType
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer, OptimizeResult
from trieste_stopping.models.trajectories import MatheronTrajectory


def get_link_function(
    model: Any, skip_identity: bool = False
) -> bijectors.Bijector | None:
    if not isinstance(model, GeneralizedGaussianProcessRegression):
        return None

    if skip_identity and isinstance(model.link_function, bijectors.Identity):
        return None

    return model.link_function


class GeneralizedGaussianProcessRegression(GaussianProcessRegression):
    """Gaussian process equivalent of a generalized linear model."""
    link_function: bijectors.Bijector

    def __init__(
        self,
        model: GPR,
        optimizer: Optimizer | None = None,
        link_function: bijectors.Bijector | None = None,
        num_kernel_samples: int = 10,
        num_rff_features: int = 1000,
        use_decoupled_sampler: bool = True,
    ):
        if link_function is None:
            link_function = bijectors.Identity()

        self.link_function = link_function
        super().__init__(
            model=model,
            optimizer=optimizer,
            num_kernel_samples=num_kernel_samples,
            num_rff_features=num_rff_features,
            use_decoupled_sampler=use_decoupled_sampler,

        )

    def get_internal_data(self, untransform: bool = True) -> Dataset:
        X, y = self.model.data
        return Dataset(X, self.link_function.inverse(y) if untransform else y)

    def optimize(self, dataset: Dataset) -> OptimizeResult:
        transformed_dataset = Dataset(
            query_points=dataset.query_points,
            observations=self.link_function(dataset.observations)
        )
        return super().optimize(dataset=transformed_dataset)

    def update(self, dataset: Dataset) -> None:
        transformed_dataset = Dataset(
            query_points=dataset.query_points,
            observations=self.link_function(dataset.observations)
        )
        return super().update(dataset=transformed_dataset)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        samples_f = self.model.predict_f_samples(query_points, num_samples)
        return self.link_function.inverse(samples_f)

    def trajectory_sampler(self, **kwargs) -> MatheronTrajectory:
        return MatheronTrajectory(
            model=self.model,  link_function=self.link_function, **kwargs
        )

