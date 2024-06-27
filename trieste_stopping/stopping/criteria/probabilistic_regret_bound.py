from __future__ import annotations

from typing import Any, Callable, Iterable

import tensorflow as tf
from gpflow.config import default_float
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box
from trieste.types import TensorType
from trieste.utils import Timer
from trieste_stopping.models import get_link_function, MatheronTrajectory
from trieste_stopping.selection import InSampleProbabilityOfMinimum, SelectionRule
from trieste_stopping.stopping.interface import (
    StoppingCriterion,
    StoppingData,
    StoppingRule,
)
from trieste_stopping.utils.level_tests import (
    _get_default_risk_schedule,
    ClopperPearsonLevelTest,
    LevelTestConvergence,
    LevelTest
)
from trieste_stopping.utils.optimization import find_start_points, run_gradient_ascent


class ProbabilisticRegretBound(StoppingRule["probabilistic_regret_bound"]):
    """
    Stopping rule from Wilson, 2024. "Stopping Bayesian Optimization with Probabilistic
    Regret Bounds".

    Stops when the simple regret of a queried point is less-equal to a threshold with
    high probability under the model.
    """

    def __init__(
        self,
        prob_bound: float,
        regret_bound: float,
        risk_bound: float | None = None,
        risk_schedule: Callable[[int], float] | None = None,
        level_test: LevelTest | None = None,
        enforce_convergence: bool = True,
        best_point_rule: SelectionRule | None = None,
    ):
        """
        Args:
            prob_bound: Upper bound on probability regret exceeds `regret_bound`.
            regret_bound: Upper bound on the simple regret.
            risk_bound: Upper bound on probability a level test emits a false result.
            risk_schedule: Schedule for the risk bound for each set of level tests.
            level_test: A sample-based test for inferring whether the probability that
                the regret exceeds `regret_bound` is above or below `prob_bound`.
            enforce_convergence: Flag controlling whether estimates produced by level
                tests that did not converge (due to resource constraints) can be used.
            best_point_rule: A rule for choosing a most-preferred point.
        """
        if risk_bound is not None:  # schedule risk bounds for each sequence of tests
            if risk_schedule is not None:
                raise ValueError("Cannot pass both `risk_bound` and `risk_schedule`.")
            risk_schedule = _get_default_risk_schedule(bound=risk_bound)

        if level_test is None:
            level_test = ClopperPearsonLevelTest(convergence=LevelTestConvergence.ANY_LE)

        super().__init__(best_point_rule=best_point_rule)
        self._level_test = level_test
        self._prob_bound = prob_bound
        self._regret_bound = regret_bound
        self._risk_schedule = risk_schedule
        self._enforce_convergence = enforce_convergence

    def prepare_stopping_criterion(
        self, model: GaussianProcessRegression, dataset: Dataset,
    ) -> probabilistic_regret_bound:
        link_function = get_link_function(model)
        sampler = RegretIndicatorSampler(
            trajectory=MatheronTrajectory(model.model, link_function=link_function),
            regret_bound=self._regret_bound,
        )

        step = float(tf.shape(dataset.observations)[0])
        return probabilistic_regret_bound(
            model=model,
            best_point_rule=self._best_point_rule,
            sampler=sampler,
            level_test=self._level_test,
            prob_bound=self._prob_bound,
            risk_bound=self._risk_schedule(step),
            enforce_convergence=self._enforce_convergence,
        )

    def update_stopping_criterion(
        self,
        criterion: probabilistic_regret_bound,
        model: GaussianProcessRegression,
        dataset: Dataset,
    ) -> probabilistic_regret_bound:
        if not isinstance(criterion, probabilistic_regret_bound):
            raise TypeError

        if model is not criterion.model:
            raise NotImplementedError

        if self._best_point_rule not in (None, criterion.best_point_rule):
            raise NotImplementedError

        # Configure the criterion
        step = float(tf.shape(dataset.observations)[0])
        criterion.prob_bound.assign(self._prob_bound)
        criterion.regret_bound.assign(self._regret_bound)
        criterion.risk_bound.assign(self._risk_schedule(step))
        criterion.enforce_convergence.assign(self._enforce_convergence)

        # Mark sampler as uninitialized
        criterion.sampler.initialized.assign(False)
        return criterion


class probabilistic_regret_bound(StoppingCriterion[GaussianProcessRegression]):
    def __init__(
        self,
        model: GaussianProcessRegression,
        sampler: RegretIndicatorSampler,
        level_test: LevelTest,
        prob_bound: float,
        risk_bound: float,
        enforce_convergence: bool = False,
        best_point_rule: SelectionRule | None = None,
        test_point_rule: SelectionRule | None = None
    ):
        super().__init__(model=model, best_point_rule=best_point_rule)
        self.sampler = sampler
        self.level_test = level_test
        self.prob_bound = tf.Variable(prob_bound, dtype=default_float())
        self.risk_bound = tf.Variable(risk_bound, dtype=default_float())
        self.enforce_convergence = tf.Variable(enforce_convergence)
        self.test_point_rule = (
                test_point_rule or InSampleProbabilityOfMinimum(num_points=5)
        )

    def __call__(self, space: Box, dataset: TensorType) -> StoppingData:
        with Timer() as timer:
            # Determine whether the criterion has been satisfied
            probs, converged = self.objective(space=space, dataset=dataset)
            if self.enforce_convergence:
                probs = tf.where(converged, probs, float("inf"))
            passed = probs <= self.prob_bound
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
            value=tf.reduce_min(tf.where(tf.math.is_nan(probs), float("inf"), probs)),
            best_point=best_point,
            setup_time=timer.time,
        )

    def objective(
        self, space: Box, dataset: Dataset, prune: bool = True
    ) -> tuple[tf.Tensor, tf.Tensor]:
        # Decide which points to test
        test_set = self.test_point_rule(self.model, space, dataset)

        # Samples indicators for whether test points fail to satisfy the regret bound
        def sampler(num_samples: TensorType) -> tf.Tensor:
            self.sampler.resample(space, num_samples)
            return tf.transpose(self.sampler(test_set.point))  # num_points x num_samples

        # Perform the level test
        estimate, (lower, upper), _ = self.level_test(
            sampler=sampler,
            level=self.prob_bound,
            risk=self.risk_bound / int(tf.shape(test_set.point)[0]),
            axis=-1,
        )
        converged = (lower >= self.prob_bound) | (upper <= self.prob_bound)
        if prune:  # fill in values that were not computed
            nan = tf.cast(float("nan"), estimate.dtype)
            size = tf.shape(dataset.query_points)[0]
            estimate, converged = (
                tf.tensor_scatter_nd_update(
                    tf.fill((size, 1), default), test_set.index[..., None], values
                ) for values, default in ((estimate, nan), (converged, False))
            )

        return tf.squeeze(estimate, axis=-1), tf.squeeze(converged, axis=-1)

    @property
    def regret_bound(self) -> tf.Variable:
        return self.sampler.regret_bound


class RegretIndicatorSampler:
    """Class for simulating whether the regret incurred by a point exceeds a level."""

    def __init__(self, trajectory: MatheronTrajectory, regret_bound: float):
        nan = float("nan")
        dtype = default_float()
        shape = tf.TensorShape(None)
        self._minima = tf.Variable(nan, dtype=dtype, shape=shape, trainable=False)
        self._minimizers = tf.Variable(nan, dtype=dtype, shape=shape, trainable=False)
        self._trajectory = trajectory
        self._initialized = tf.Variable(False, trainable=False)
        self.regret_bound = tf.Variable(regret_bound, dtype=dtype, trainable=False)

    def __call__(self, x: TensorType) -> tf.Tensor:
        if not self.initialized:
            raise RuntimeError("Please use `resample` first to initialize.")

        samples = tf.squeeze(self.trajectory(x), -1)
        return tf.greater(samples, self.minima + self.regret_bound)

    def resample(self, space: Box, num_samples: int, **kwargs: Any) -> None:
        self._initialized.assign(False)  # mark as uninitialized

        # Sample new trajectories
        self.trajectory.resample([num_samples])

        # Find/assign new minimizers and minima
        minimizers, minima = self._minimize(space, **kwargs)
        self._minimizers.assign(minimizers)
        self._minima.assign(minima)
        self._initialized.assign(True)  # mark as initialized

    def _minimize(
        self,
        space: Box,
        num_starts: int = 16,
        num_random_batches: int = 64,
        random_batch_size: int = 256,
        custom_batches: Iterable[TensorType] | None = None,
        scipy_kwargs: dict | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Helper method for finding pathwise minimizers and minima."""
        if tf.rank(self.trajectory.batch_shape) != 1:
            raise NotImplementedError(
                f"Expected rank-1 batch shape but {self.trajectory.batch_shape=}"
            )

        # Find starting points for each trajectory
        start_points, _ = find_start_points(
            fun=self._ascent_objective,
            space=space,
            num_starts=num_starts,
            custom_batches=custom_batches,
            num_random_batches=num_random_batches,
            random_batch_shape=(self.trajectory.batch_shape[0], random_batch_size),
        )

        # Run gradient descent from each trajectory-specific starting point
        return self._run_gradient_descent(space, start_points, scipy_kwargs)

    def _run_gradient_descent(
        self, space: Box, start_points: TensorType, scipy_kwargs: dict | None = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        final_points, _ = run_gradient_ascent(
            fun=self._ascent_objective,
            space=space,
            start_points=start_points,
            scipy_kwargs=scipy_kwargs,
        )
        final_values = self.trajectory(final_points)

        # Return the best point and value for each trajectory
        indices = tf.argmin(tf.squeeze(final_values, -1), axis=-1, output_type=tf.int32)
        best_points = tf.gather(final_points, indices, axis=-2, batch_dims=1)
        best_values = tf.gather(final_values, indices, axis=-2, batch_dims=1)
        return best_points, best_values

    @tf.function
    def _ascent_objective(self, x: TensorType) -> TensorType:
        return -tf.squeeze(self.trajectory(x), axis=-1)

    @property
    def trajectory(self) -> MatheronTrajectory:
        return self._trajectory

    @property
    def minima(self) -> TensorType:
        return self._minima

    @property
    def minimizers(self) -> TensorType:
        return self._minimizers

    @property
    def initialized(self) -> tf.Variable:
        return self._initialized
