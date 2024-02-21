from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Iterable
from warnings import warn

import tensorflow as tf
from gpflow.config import default_float
from tensorflow_probability.python.internal.special_math import ndtr
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box
from trieste.types import TensorType
from trieste.utils import Timer
from trieste_stopping.incumbent import IncumbentRule
from trieste_stopping.models.trajectories import MatheronTrajectory
from trieste_stopping.stopping.interface import (
    StoppingCriterion,
    StoppingData,
    StoppingRule,
)
from trieste_stopping.utils.optimization import find_start_points, run_gradient_ascent
from trieste_stopping.utils.probability import (
    adaptive_empirical_bernstein_estimator,
    AdaptiveEstimatorConfig,
    build_power_schedule,
    EstimatorStoppingCondition,
)


class ProbabilisticRegretBound(StoppingRule["probabilistic_regret_bound"]):
    def __init__(
        self,
        regret_bound: float,
        risk_tolerance_model: float,
        risk_tolerance_error: float | Callable[[int], float],
        time_limit: float = float("inf"),
        batch_limit: int = 256,
        popsize_limit: int = 2 ** 31 - 1,
        use_unconverged_estimates: bool = False,
        incumbent_rule: IncumbentRule | None = None,
    ):
        if time_limit == float("inf") and popsize_limit == 2 ** 31 - 1:
            warn(
                "No limits were placed on the amount of time or number of samples the"
                " estimator is allowed to use. This may result in exceedingly long"
                " runtimes when evaluating the stopping rule."
            )

        super().__init__(incumbent_rule=incumbent_rule)
        self.estimator_config = AdaptiveEstimatorConfig(
            time_limit=time_limit,
            batch_limit=batch_limit,
            popsize_limit=popsize_limit,
            stopping_condition=EstimatorStoppingCondition.ANY_LE,
        )
        self.regret_bound = regret_bound
        self.risk_tolerance_model = risk_tolerance_model
        self.risk_tolerance_error_schedule = (
            risk_tolerance_error
            if isinstance(risk_tolerance_error, Callable)
            else build_power_schedule(risk_tolerance_error)
        )
        self.use_unconverged_estimates = use_unconverged_estimates

    def prepare_stopping_criterion(
        self, model: GaussianProcessRegression, dataset: Dataset,
    ) -> probabilistic_regret_bound:
        step = int(tf.shape(dataset.observations)[0])
        sampler = RegretIndicatorSampler(
            trajectory=MatheronTrajectory(model=model.model),
            regret_bound=self.regret_bound,
        )
        return probabilistic_regret_bound(
            model=model,
            incumbent_rule=self.incumbent_rule,
            sampler=sampler,
            estimator_config=self.estimator_config,
            risk_tolerance_model=self.risk_tolerance_model,
            risk_tolerance_error=self.risk_tolerance_error_schedule(step),
            use_unconverged_estimates=self.use_unconverged_estimates,
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

        if self.incumbent_rule not in (None, criterion.incumbent_rule):
            raise NotImplementedError

        # Configure the criterion
        step = int(tf.shape(dataset.observations)[0])
        criterion.estimator_config = self.estimator_config
        criterion.use_unconverged_estimates = self.use_unconverged_estimates

        criterion.regret_bound.assign(self.regret_bound)
        criterion.risk_tolerance_model.assign(self.risk_tolerance_model)
        criterion.risk_tolerance_error.assign(self.risk_tolerance_error_schedule(step))

        # Mark sampler as uninitialized
        criterion.sampler.initialized.assign(False)

        return criterion


class probabilistic_regret_bound(
    StoppingCriterion[GaussianProcessRegression]
):
    def __init__(
        self,
        model: GaussianProcessRegression,
        sampler: RegretIndicatorSampler,
        risk_tolerance_model: float,
        risk_tolerance_error: float,
        estimator_config: AdaptiveEstimatorConfig,
        use_unconverged_estimates: bool = False,
        incumbent_rule: IncumbentRule | None = None,
    ):
        super().__init__(model=model, incumbent_rule=incumbent_rule)
        self.sampler = sampler
        self.estimator_config = estimator_config
        self.use_unconverged_estimates = use_unconverged_estimates

        dtype = default_float()
        self.risk_tolerance_model = tf.Variable(risk_tolerance_model, dtype=dtype)
        self.risk_tolerance_error = tf.Variable(risk_tolerance_error, dtype=dtype)

    def __call__(self, space: Box, dataset: TensorType) -> StoppingData:
        with Timer() as timer:
            # Determine whether the criterion has been satisfied
            estimates, converged = self.evaluate(space=space, dataset=dataset)
            risk = (
                estimates
                if self.use_unconverged_estimates
                else tf.where(converged, estimates, float("inf"))
            )
            mask = risk <= self.risk_tolerance_model
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
            value=tf.reduce_min(tf.where(tf.math.is_nan(risk), float("inf"), risk)),
            incumbent=incumbent,
            setup_time=timer.time,
        )

    def evaluate(
        self, space: Box, dataset: Dataset, prune: bool = True
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if prune:
            test_mask = self.prune_test_points(points=dataset.query_points)
            test_points = dataset.query_points[test_mask]
        else:
            test_points = dataset.query_points

        # Samples indicators for whether test points fail to satisfy the regret bound
        def _sampler(num_samples: TensorType) -> tf.Tensor:
            self.sampler.resample(space, num_samples)
            failure_indicators = self.sampler(test_points)
            return tf.transpose(tf.cast(failure_indicators, tf.float64))

        # Run the adaptive estimator
        num_tests = int(tf.shape(test_points)[0])
        moments, error_bounds = adaptive_empirical_bernstein_estimator(
            sampler=_sampler,
            sample_width=1.0,
            threshold=self.risk_tolerance_model,
            risk_tolerance=self.risk_tolerance_error / num_tests,
            **asdict(self.estimator_config)
        )

        estimates = moments.mean
        converged = tf.abs(estimates - self.risk_tolerance_model) >= error_bounds

        if prune:  # fill in values that were not computed
            nan = tf.cast(float("nan"), estimates.dtype)
            size = tf.shape(dataset.query_points)[0]
            estimates, converged = (
                tf.squeeze(
                    tf.tensor_scatter_nd_update(
                        tf.fill((size, 1), default),
                        tf.where(test_mask),
                        tf.expand_dims(values, -1)
                    ), axis=-1,
                ) for values, default in ((estimates, nan), (converged, False))
            )

        return estimates, converged

    def prune_test_points(self, points: TensorType) -> TensorType:
        """
        Uses an upper bound on the probability that `E = f(x) <= min f + epsilon`
        to prune points incapable of satisfying the criterion `P(E) > 1 - delta`
        """
        # Evaluate joint posterior for `f(X)`
        mean, covariance = self.model.predict_joint(points)
        covariance = tf.squeeze(covariance, axis=0)
        variances = tf.linalg.diag_part(covariance)
        incumbent = tf.squeeze(tf.argmin(mean, axis=-2), -1)  # index of best point

        # Compute means and variances for `f(x) - f(x*)`
        d_means = tf.squeeze(mean - tf.gather(mean, incumbent, axis=-2), axis=-1)
        d_variances = (
                variances
                + tf.gather(variances, incumbent, axis=-1)
                - 2 * tf.gather(covariance, incumbent, axis=-1)
        )
        d_variances = tf.maximum(d_variances, 1e-12)  # clip for numerical stability

        # Calculate complements of upper bounds `P[f(x) - f(x*) <= epsilon]`
        complements = ndtr(tf.math.rsqrt(d_variances) * (d_means - self.regret_bound))
        accept = tf.tensor_scatter_nd_update(  # ensure index point is accepted
            tensor=tf.expand_dims(complements <= self.risk_tolerance_model, -1),
            indices=[[incumbent]],
            updates=[[True]],
        )
        return tf.squeeze(accept, axis=-1)

    @property
    def regret_bound(self) -> tf.Variable:
        return self.sampler.regret_bound


class RegretIndicatorSampler:
    def __init__(
        self, trajectory: MatheronTrajectory, regret_bound: float | TensorType
    ):
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
            raise RuntimeError

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
