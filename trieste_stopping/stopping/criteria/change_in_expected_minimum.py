from __future__ import annotations

from math import pi

import tensorflow as tf
from tensorflow.python.ops.numpy_ops.np_math_ops import allclose, diff
from tensorflow_probability.python.internal.special_math import ndtr
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box
from trieste.types import TensorType
from trieste.utils import Timer
from trieste_stopping.models import get_link_function
from trieste_stopping.utils import get_expected_value
from trieste_stopping.stopping.criteria.confidence_bound import (
    confidence_bound,
    ConfidenceBound
)
from trieste_stopping.stopping.interface import StoppingData


class ChangeInExpectedMinimum(ConfidenceBound):
    """
    Stopping rule from Ishibashi et al., 2023. "A stopping criterion for
    Bayesian optimization by the gap of expected minimum simple regrets".

    Stops when an upper bound on the absolute difference between expected
    minima at times `T - 1` and `T` is less-equal to a threshold.
    """

    def prepare_stopping_criterion(
        self, model: GaussianProcessRegression, dataset: Dataset,
    ) -> change_in_expected_minimum:
        if self._beta_schedule is None:
            self._beta_schedule = self._get_default_beta_schedule(
                dataset.query_points.shape[-1]
            )

        step = int(tf.shape(dataset.observations)[0])
        return change_in_expected_minimum(
            model=model,
            best_point_rule=self._best_point_rule,
            threshold=self._threshold,
            beta=self._beta_schedule(step),
        )


class change_in_expected_minimum(confidence_bound):
    def __call__(self, space: Box, dataset: TensorType) -> StoppingData:
        with Timer() as timer:
            bound = tf.squeeze(self.objective(space=space, dataset=dataset))
            best_point = self.best_point_rule(
                model=self.model, space=space, dataset=dataset
            )

        return StoppingData(
            done=bound < self.threshold,
            value=bound,
            best_point=best_point,
            setup_time=timer.time,
        )

    def objective(self, space: Box, dataset: Dataset) -> tf.Tensor:
        """
        Evaluates an upper bound on the absolute difference in the expected value
        of the minima at times `T - 1` and `T`. See Ishibashi et al., 2023.

        Args:
            space: The space over which to compute the bound.
            dataset: A dataset consisting of `T` labelled pairs.

        Returns: A scalar.
        """
        internal_dataset = self.model.get_internal_data()  # check that the data matches
        if not (
            allclose(dataset.query_points, internal_dataset.query_points)
            and allclose(dataset.observations, internal_dataset.observations)
        ):
            raise NotImplementedError

        X, xT = tf.split(dataset.query_points, [-1, 1], axis=0)
        Y, yT = tf.split(dataset.observations, [-1, 1], axis=0)
        old_dataset = Dataset(X, Y)

        # Calculate old regret bound and KL divergence between old and new posteriors
        self.model.update(old_dataset)
        bound = tf.reduce_min(super().objective(space, dataset=old_dataset))  # UCB bound
        mT, vT = self.model.predict(xT)
        noise2 = self.model.model.likelihood.variance
        bound *= 0.5 * tf.sqrt(  # multiply by sqrt(0.5 * KL)
            tf.math.log(1 + vT / noise2)
            - vT / (vT + noise2)
            + vT * tf.square((yT - mT) / (vT + noise2))
        )

        # Determine old and new best points
        old_best = self.best_point_rule(self.model, space, dataset=old_dataset)
        self.model.update(dataset)  # restore the model
        new_best = self.best_point_rule(self.model, space, dataset=dataset)

        # Add absolute difference in expected value of best points under both posteriors
        link_function = get_link_function(self.model, skip_identity=True)
        expected_vals = (
            get_expected_value(p.mean, p.variance, inverse=link_function)
            for p in (new_best, old_best)
        )
        bound += tf.abs(tf.subtract(*expected_vals))

        # Maybe terminate early if the EI term will be zero
        if tf.reduce_all(old_best.point == new_best.point):
            return bound

        # Add expected improvement of old over new best points under current posterior
        points = tf.stack([new_best.point, old_best.point], axis=-2)
        if link_function is None:
            m, K = self.model.predict_joint(points)
            K = tf.squeeze(K, axis=0)  # squeeze explicit output dimension
            m = diff(m, axis=-2)  # mean and stddev of f(new) - f(old)
            s = tf.sqrt(K[0, 0] - 2 * K[0, 1] + K[1, 1])
            u = m / s
            bound += tf.where(
                s > 0,
                m * ndtr(u) + s * ((2 * pi) ** -0.5) * tf.exp(-0.5 * u ** 2),
                0.0
            )
        else:
            draws = self.model.sample(points, num_samples=4096)  # inverse linked!
            diffs = diff(draws, axis=-2)
            bound += tf.reduce_mean(tf.maximum(diffs, 0.0), axis=0)

        return bound
