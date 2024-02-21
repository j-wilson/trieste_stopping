from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import SearchSpace
from trieste.types import TensorType


@dataclass
class IncumbentData:
    point: TensorType
    observation: TensorType | None = None
    index: int | None = None
    mean: TensorType | None = None
    variance: TensorType | None = None


class IncumbentRule(ABC):
    """Interface for a generic incumbent rule."""
    @abstractmethod
    def __call__(
        self,
        model: Any,
        space: SearchSpace,
        dataset: Dataset,
        **kwargs: Any,
    ) -> IncumbentData:
        pass


class InSamplePosteriorMinimizer(IncumbentRule):
    """Returns the dataset entry that minimizes the posterior mean."""
    def __call__(
        self,
        model: GaussianProcessRegression,
        space: SearchSpace,
        dataset: Dataset,
        **kwargs: Any,
    ) -> IncumbentData:
        means, variances = model.predict(dataset.query_points)
        index = tf.argmin(tf.squeeze(means, axis=-1))
        return IncumbentData(
            point=dataset.query_points[index],
            observation=(
                None if dataset.observations is None else dataset.observations[index]
            ),
            index=index,
            mean=means[index],
            variance=variances[index],
        )
