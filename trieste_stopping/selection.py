from __future__ import annotations

from abc import ABC, abstractmethod
from math import log, pi
from typing import Any

import tensorflow as tf
from numpy.polynomial.hermite import hermgauss
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import SearchSpace
from trieste_stopping.models import get_link_function
from trieste_stopping.utils import get_expected_value, PointData
from tensorflow_probability.python.bijectors import Bijector
from tensorflow_probability.python.internal.special_math import log_ndtr, ndtr


class SelectionRule(ABC):
    """Interface for a generic method for selecting a set of points."""
    @abstractmethod
    def __call__(
        self,
        model: Any,
        space: SearchSpace,
        dataset: Dataset,
        **kwargs: Any,
    ) -> PointData:
        pass


class InSamplePosteriorMinimizer(SelectionRule):
    """Returns a point whose expected value under the posterior is the smallest."""
    def __call__(
        self,
        model: GaussianProcessRegression,
        space: SearchSpace,
        dataset: Dataset,
        **kwargs: Any,
    ) -> PointData:
        link_function = get_link_function(model)
        means, variances = model.predict(dataset.query_points)
        expected_values = get_expected_value(means, variances, inverse=link_function)
        index = tf.argmin(tf.squeeze(expected_values, axis=-1), axis=0)
        return PointData(
            point=tf.gather(dataset.query_points, index),
            index=index,
            mean=tf.gather(means, index),
            variance=tf.gather(variances, index),
            observation=(
                None
                if dataset.observations is None
                else tf.gather(dataset.observations, index)
            ),
        )


class GreedySelectionRule(SelectionRule):
    """Base class for selection rules that greedily construct a set."""
    def __init__(self, num_points: int, num_samples: int = 256):
        if num_points < 1:
            raise ValueError

        self.num_points = num_points
        self.num_samples = num_samples

    def __call__(
        self,
        model: GaussianProcessRegression,
        space: SearchSpace,
        dataset: Dataset,
        **kwargs: Any,
    ) -> PointData:
        # Initialize
        link_function = get_link_function(model, skip_identity=True)
        means, covariance = model.predict_joint(dataset.query_points)
        covariance = tf.squeeze(covariance, 0)  # squeeze explicit output dimension
        initial_means = tf.identity(means)  # retain a copy of these
        initial_variances = tf.expand_dims(tf.linalg.diag_part(covariance), -1)

        # Select first point
        active = [index for index in range(tf.shape(means)[-2])]
        choice = self.select_first(initial_means, initial_variances, link_function)
        chosen = [active.pop(choice)]
        minima = float("inf")

        # Greedily select additional points
        while active and len(chosen) < self.num_points:
            # Sample values for the previously chosen point and update terms
            idx = slice(choice, choice + 1)
            xov = covariance[:, idx]
            vec = tf.math.rsqrt(xov[idx]) * xov
            means += vec * tf.random.normal([self.num_samples, 1, 1], dtype=vec.dtype)
            covariance -= tf.matmul(vec, vec, transpose_b=True)
            minima = tf.minimum(minima, means[:, idx, :])  # means[:, idx, :] := samples

            # Drop the previously chosen term
            keep = [index for index in range(tf.shape(means)[-2]) if index != choice]
            means = tf.gather(means, keep, axis=-2)
            covariance = tf.gather(tf.gather(covariance, keep, axis=-2), keep, axis=-1)

            # Choose next point
            choice = self.select_next(
                means=means,
                variances=tf.expand_dims(tf.linalg.diag_part(covariance), -1),
                minima=minima,
                link_function=link_function,
            )
            if choice is None:
                break

            chosen.append(active.pop(choice))  # store index and remove from active set

        return PointData(
            point=tf.gather(dataset.query_points, chosen),
            index=tf.stack(chosen),
            mean=tf.gather(initial_means, chosen),
            variance=tf.gather(initial_variances, chosen),
            observation=(
                None
                if dataset.observations is None
                else tf.gather(dataset.observations, chosen)
            ),
        )

    @abstractmethod
    def select_next(
        self,
        means: tf.Tensor,
        variances: tf.Tensor,
        minima: tf.Tensor,
        link_function: Bijector | None = None,
    ) -> int | None:
        """
        Args:
            means: A tensor of means with shape [num_samples, n, 1].
            variances: A tensor of variances with shape [n, 1].
            minima: A tensor of sample minima with shape [num_samples, n, 1].
            link_function: An optional link function.

        Returns: A optional index for the selected point.
        """

    def select_first(
        self,
        means: tf.Tensor,
        variances: tf.Tensor,
        link_function: Bijector | None = None,
    ) -> int:
        expected_values = get_expected_value(means, variances, inverse=link_function)
        return tf.argmin(tf.squeeze(expected_values, -1))


class InSampleExpectedMinimum(GreedySelectionRule):
    """
    Generalization of InSamplePosteriorMinimizer. Greedily constructs $A \subset X$
    to approximately minimizes the expected min, i.e. $E[\min_{x \in A} f(x)]$.
    """
    def __init__(self, num_points: int, num_samples: int = 256, cutoff: float = 0.0):
        if num_points < 1:
            raise ValueError

        self.cutoff = cutoff
        self.num_points = num_points
        self.num_samples = num_samples

    def select_next(
        self,
        means: tf.Tensor,
        variances: tf.Tensor,
        minima: tf.Tensor,
        link_function: Bijector | None = None,
    ) -> int | None:
        """Returns the index of the point that maximally reduces the expected min."""
        # For each sample, compute decrease in the expected min
        s = tf.sqrt(variances)
        if link_function is None:  # closed-form
            s = tf.squeeze(s, -1)
            m = tf.squeeze(minima - means, -1)
            u = m / s
            ei_per_sample = tf.where(
                s > 0,
                s * ((2 * pi) ** -0.5) * tf.exp(-0.5 * u ** 2) + m * ndtr(u),
                0.0
            )
        else:  # quadrature-based estimates
            z, dz = (tf.convert_to_tensor(arr, dtype=s.dtype) for arr in hermgauss(16))
            draws = means + (2 ** 0.5) * s * z
            diffs = link_function.inverse(minima) - link_function.inverse(draws)
            ei_per_sample = (
                (pi ** -0.5) * tf.reduce_sum(tf.nn.relu(diffs) * dz, -1)
            )

        ei = tf.reduce_mean(ei_per_sample, 0)
        return tf.argmax(ei) if tf.reduce_any(ei > self.cutoff) else None


class InSampleProbabilityOfMinimum(GreedySelectionRule):
    """
    Greedily constructs a subset $A \subset X$ that approximately maximizes the
    probability of containing a minimizer, i.e. $P(argmin_{x \in X} f(x) \in A)$.
    """
    def __init__(self, num_points: int, num_samples: int = 256, cutoff: float = 0.0):
        if num_points < 1:
            raise ValueError

        self.cutoff = cutoff
        self.num_points = num_points
        self.num_samples = num_samples

    def select_next(
        self,
        means: tf.Tensor,
        variances: tf.Tensor,
        minima: tf.Tensor,
        link_function: Bijector | None = None,
    ) -> int | None:
        """
        Returns the index of the point that maximally increases the probability that
        the enlarged subset contains a minimizer.
        """
        # For each sample, compute probabilty that each RVs is less than the current min
        if link_function is None:
            temp = tf.squeeze(tf.math.rsqrt(variances) * (minima - means), -1)
            temp = tf.where(tf.squeeze(variances, -1) > 0, temp, float("-inf"))
            probs = tf.reduce_logsumexp(log_ndtr(temp), 0) - log(self.num_samples)
            cutoff = tf.math.log(tf.cast(self.cutoff, probs.dtype))
            return tf.argmax(probs) if tf.reduce_any(probs > cutoff) else None

        # Quadrature-based estimates
        z, dz = (tf.convert_to_tensor(arr, dtype=means.dtype) for arr in hermgauss(16))
        draws = means + tf.sqrt(2 * variances) * z
        flags = link_function.inverse(draws) < link_function.inverse(minima)
        if not tf.reduce_any(flags):
            return None

        probs = tf.reduce_mean(
            (pi ** -0.5) * tf.linalg.matvec(tf.cast(flags, dtype=dz.dtype), dz), 0
        )
        return tf.argmax(probs) if tf.reduce_any(probs > self.cutoff) else None
