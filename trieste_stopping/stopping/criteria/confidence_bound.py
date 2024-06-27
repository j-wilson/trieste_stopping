from __future__ import annotations

import tensorflow as tf
from gpflow.config import default_float
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box, SearchSpace
from trieste.types import TensorType
from trieste.utils import Timer
from trieste_stopping.selection import SelectionRule
from trieste_stopping.models.models import GeneralizedGaussianProcessRegression
from trieste_stopping.utils import run_multistart_gradient_ascent
from trieste_stopping.utils.schedules import BetaParameterSchedule, Schedule
from trieste_stopping.stopping.interface import (
    StoppingCriterion,
    StoppingData,
    StoppingRule,
)


class ConfidenceBound(StoppingRule["confidence_bound"]):
    """
    Stopping rule from Makarova et al., 2022. "Automatic Termination for Hyperparameter
    Optimization".

    Stops when the difference between upper and lower confidence bounds is less-equal to
    a threshold. For appropriate chosen schedules of `beta`, this correspond to a
    high-probabililty upper bound on the simple regret.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        risk_bound: float | None = None,
        beta_schedule: Schedule[float] | None = None,
        best_point_rule: SelectionRule| None = None,
    ):
        """
        Args:
            threshold: A scalar used to decide when to stop.
            risk_bound: Upper bound on probability used to define confidence bounds.
            beta_schedule: A schedule for the beta parameter.
            best_point_rule: A rule for choosing a most-preferred point.
        """
        if not (risk_bound is None) ^ (beta_schedule is None):
            raise ValueError(
                "One and only one of `risk_bound` or `beta_schedule` must be passed."
            )

        super().__init__(best_point_rule=best_point_rule)
        self._threshold = threshold
        self._risk_bound = risk_bound
        self._beta_schedule = beta_schedule

    def prepare_stopping_criterion(
        self, model: GaussianProcessRegression, dataset: Dataset,
    ) -> confidence_bound:
        if self._beta_schedule is None:
            self._beta_schedule = self._get_default_beta_schedule(
                dataset.query_points.shape[-1]
            )

        step = int(tf.shape(dataset.observations)[0])
        return confidence_bound(
            model=model,
            best_point_rule=self._best_point_rule,
            threshold=self._threshold,
            beta=self._beta_schedule(step),
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

        if self._best_point_rule not in (None, criterion.best_point_rule):
            raise NotImplementedError

        if self._beta_schedule is None:
            self._beta_schedule = self._get_default_beta_schedule(
                dataset.query_points.shape[-1]
            )

        step = int(tf.shape(dataset.observations)[0])
        criterion.threshold.assign(self._threshold)
        criterion.beta.assign(self._beta_schedule(step))
        return criterion

    def _get_default_beta_schedule(self, dim: int, scale: float = 2/5
    ) -> BetaParameterSchedule:
        """Returns the schedule for the beta parameter from Makarova et al., 2022."""
        return BetaParameterSchedule(dim, risk_bound=self._risk_bound, scale=2 / 5)


class confidence_bound(StoppingCriterion[GaussianProcessRegression]):
    def __init__(
        self,
        model: GaussianProcessRegression,
        threshold: float,
        beta: float,
        best_point_rule: SelectionRule | None = None,
    ):
        super().__init__(model=model, best_point_rule=best_point_rule)
        self.beta = tf.Variable(beta, dtype=default_float())
        self.threshold = tf.Variable(threshold, dtype=default_float())

    def __call__(self, space: SearchSpace, dataset: Dataset) -> StoppingData:
        with Timer() as timer:
            bounds = self.objective(space=space, dataset=dataset)
            passed = bounds <= self.threshold
            done = tf.reduce_any(passed)

            # Choose a best point from the candidate set
            candidate_data = (
                Dataset(dataset.query_points[passed], dataset.observations[passed])
                if done
                else dataset
            )
            best_point = self.best_point_rule(
                model=self.model,
                space=space,
                dataset=candidate_data,
            )
            if done:  # replace with global index
                best_point.index = tf.squeeze(tf.where(passed)[best_point.index])

        return StoppingData(
            done=done,
            value=tf.reduce_min(bounds),
            best_point=best_point,
            setup_time=timer.time,
        )

    def objective(self, space: Box, dataset: Dataset) -> tf.Tensor:
        """
        Evaluates `UCB(query_points) - min_{x \in space} LCB(x)`.

        Args:
            space: The space over which to compute the bound.
            dataset: A dataset consting of labelled pairs.

        Returns: A vector of bounds.
        """
        ucb_values = self._upper_confidence_bound(dataset.query_points)
        nlcb_point, nlcb_value = run_multistart_gradient_ascent(
            fun=self._neg_lower_confidence_bound,
            space=space,
            num_cmaes_runs=1 if space.dimension > 1 else 0,
        )
        return ucb_values + nlcb_value

    @tf.function
    def _neg_lower_confidence_bound(self, x: TensorType) -> TensorType:
        mean, variance = self.model.predict(x)
        bound = mean - tf.sqrt(variance * self.beta)
        if isinstance(self.model, GeneralizedGaussianProcessRegression):
            bound = self.model.link_function.inverse(bound)  # assumed increasing
        return -tf.squeeze(bound, axis=-1)

    @tf.function
    def _upper_confidence_bound(self, x: TensorType) -> TensorType:
        mean, variance = self.model.predict(x)
        bound = mean + tf.sqrt(variance * self.beta)
        if isinstance(self.model, GeneralizedGaussianProcessRegression):
            bound = self.model.link_function.inverse(bound)  # assumed increasing
        return tf.squeeze(bound, axis=-1)
