from __future__ import annotations

from math import log, pi
from typing import Callable

import tensorflow as tf
from gpflow.config import default_float
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box
from trieste.types import TensorType
from trieste.utils import Timer
from trieste_stopping.incumbent import IncumbentRule
from trieste_stopping.utils import run_multistart_gradient_ascent
from trieste_stopping.stopping.interface import (
    StoppingCriterion,
    StoppingData,
    StoppingRule,
)


class ConfidenceBound(StoppingRule["confidence_bound"]):
    def __init__(
        self,
        regret_bound: float,
        beta_schedule: Callable[[int], float],
        incumbent_rule: IncumbentRule | None = None,
):
        super().__init__(incumbent_rule=incumbent_rule)
        self.regret_bound = regret_bound
        self.beta_schedule = beta_schedule

    def prepare_stopping_criterion(
        self, model: GaussianProcessRegression, dataset: Dataset,
    ) -> confidence_bound:
        step = int(tf.shape(dataset.observations)[0])
        return confidence_bound(
            model=model,
            incumbent_rule=self.incumbent_rule,
            regret_bound=self.regret_bound,
            beta=self.beta_schedule(step),
        )

    def update_stopping_criterion(
        self,
        criterion: confidence_bound,
        model: GaussianProcessRegression,
        dataset: Dataset,
    ) -> confidence_bound:
        if not isinstance(criterion, confidence_bound):
            raise TypeError

        if model is not criterion.model:
            raise NotImplementedError

        if self.incumbent_rule not in (None, criterion.incumbent_rule):
            raise NotImplementedError

        step = int(tf.shape(dataset.observations)[0])
        criterion.regret_bound.assign(self.regret_bound)
        criterion.beta.assign(self.beta_schedule(step))
        return criterion


class confidence_bound(StoppingCriterion[GaussianProcessRegression]):
    def __init__(
        self,
        model: GaussianProcessRegression,
        regret_bound: float,
        beta: float,
        incumbent_rule: IncumbentRule | None = None,
    ):
        super().__init__(model=model, incumbent_rule=incumbent_rule)
        self.beta = tf.Variable(beta, dtype=default_float())
        self.regret_bound = tf.Variable(regret_bound, dtype=default_float())

    def __call__(self, space: Box, dataset: TensorType) -> StoppingData:
        with Timer() as timer:
            bounds = self.evaluate(space=space, dataset=dataset)
            mask = bounds <= self.regret_bound
            done = tf.reduce_any(mask)
            if done:  # evaluate the incumbent rule on points that satisfy the criterion
                incumbent = self.incumbent_rule(
                    model=self.model,
                    space=space,
                    dataset=Dataset(dataset.query_points[mask], dataset.observations[mask]),
                )
                incumbent.index = tf.squeeze(tf.where(mask)[incumbent.index])
            else:  # evaluate the incumbent rule on all points
                incumbent = self.incumbent_rule(
                    model=self.model, space=space, dataset=dataset
                )

        return StoppingData(
            done=done,
            value=tf.reduce_min(bounds),
            incumbent=incumbent,
            setup_time=timer.time,
        )

    def evaluate(self, space: Box, dataset: Dataset) -> tf.Tensor:
        """
        Tests whether `min UCB(query_points) - min LCB(space) <= regret_bound`.
        """
        with Timer() as timer:
            ucb_values = self._upper_confidence_bound(dataset.query_points)
            threshold = self.regret_bound - tf.reduce_min(ucb_values)
            nlcb_point, nlcb_value = run_multistart_gradient_ascent(
                fun=self._neg_lower_confidence_bound,
                space=space,
                num_cmaes_runs=1 if space.dimension > 1 else 0,
                scipy_kwargs={"stop_callback": lambda res: res.fun >= threshold},
            )
            bounds = ucb_values + nlcb_value

        return bounds

    @tf.function
    def _neg_lower_confidence_bound(self, x: TensorType) -> TensorType:
        mean, variance = self.model.predict(x)
        return tf.squeeze(tf.sqrt(variance * self.beta) - mean, -1)

    def _upper_confidence_bound(self, x: TensorType) -> TensorType:
        mean, variance = self.model.predict(x)
        return tf.squeeze(mean + tf.sqrt(variance * self.beta), -1)


def build_default_beta_schedule(risk_tolerance: float, dim: int):
    """Helper method for the beta schedule recommended by Makarova et. al, 2022."""
    def schedule_func(step: int) -> float:
        return (2 / 5) * log((step * pi) ** 2 * dim / (6 * risk_tolerance))
    return schedule_func
