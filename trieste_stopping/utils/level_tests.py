from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Callable, NamedTuple
from warnings import warn

import tensorflow as tf
from gpflow.base import default_float
from tensorflow_probability.python.math import find_root_chandrupatla
from tensorflow_probability.python.distributions import Beta
from tensorflow_probability.python.distributions.binomial import _bdtr
from trieste.types import TensorType
from trieste_stopping.utils.schedules import GeometricSchedule, PowerSchedule

_MAXINT = 2 ** 31 - 1


def _get_dtype(x: int | float | tf.Tensor) -> tf.DType:
    return x.dtype if tf.is_tensor(x) else default_float()


def _get_default_risk_schedule(bound: float | tf.Tensor) -> PowerSchedule:
    """Returns a function f(t) -> R_{+} such that \sum_{t=1}^{\infty} f(t) <= limit"""
    return PowerSchedule(-1.1, initial_value=(0.1 / 1.1) * bound)


class ConfidenceInterval(NamedTuple):
    lower: TensorType
    upper: TensorType


class LevelTestResult(NamedTuple):
    estimate: TensorType
    confidence_interval: ConfidenceInterval
    sample_size: int


class LevelTestConvergence(str, Enum):
    """Controls the estimator's stopping behavior when tests are run in parallel."""
    ALL = "all"  # stop when all (confidence) intervals exclude the level
    ANY = "any"  # stop when any interval excludes the level
    # Stop when all intervals exclude the level OR...
    ANY_GE = "any_ge"  # any interval excludes the level from above
    ANY_LE = "any_le"  # any interval excludes the level from below


class LevelTest:
    """
    Performs a sequence of sample-based tests to determine whether a random variable's
    expected value is above or below a level with high probability.
    """
    def __init__(
        self,
        step_limit: int | None = None,
        size_limit: int = _MAXINT,
        time_limit: float = float("inf"),
        batch_limit: int = _MAXINT,
        convergence: LevelTestConvergence = LevelTestConvergence.ALL,
        size_schedule: Callable[[int], int] | None = None,
    ):
        """
        Args:
            step_limit: Upper bound on the number of tests.
            size_limit: Upper bound on the number of draws.
            time_limit: Upper bound on the total wall time.
            batch_limit: Upper bound on the number of draws generated simultaneously.
            convergence: Convergence criterion; defaults to all tests to converging.
            size_schedule: Custom schedule for the number of samples used by each test;
                defaults to a GeometricSchedule.
        """
        if time_limit == float("inf") and size_limit >= _MAXINT:
            warn(
                "No limits were set for the amount of time or number of samples that"
                " tests may use. This may result in excessively long runtimes."
            )

        if size_schedule is None:
            size_schedule = GeometricSchedule(1.5, initial_value=32)

        self.size_limit = size_limit
        self.step_limit = _MAXINT if step_limit is None else step_limit
        self.time_limit = time_limit
        self.batch_limit = batch_limit
        self.size_schedule = size_schedule
        self.convergence = convergence

    @abstractmethod
    def __call__(
        self,
        sampler: Callable[[int], tf.Tensor],
        level: float | tf.Tensor,
        risk: float | Callable[[int], float],
        axis: int,
    ) -> LevelTestResult:
        """
        Main method for executing tests.

        Args:
            sampler: Function used to generate samples.
            level: Level against which to compare expected values.
            risk: Upper bound on the probability of a false positive or false negative.
            axis: Dimension along which to reduce sample values.

        Returns: A LevelTestResult object.
        """

    def check_stopping(
        self,
        step: int,
        size: int,
        level: tf.Tensor,
        lower: tf.Tensor,
        upper: tf.Tensor,
        cutoff_time: float = float("inf"),
    ) -> bool:
        """Evaluates whether to terminate the sequence of tests."""
        if self.convergence is LevelTestConvergence.ALL:
            converged = tf.reduce_all((level <= lower) | (level >= upper))
        elif self.convergence is LevelTestConvergence.ANY:
            converged = tf.reduce_any((level <= lower) | (level >= upper))
        elif self.convergence is LevelTestConvergence.ANY_GE:
            converged = tf.reduce_any(level <= lower) | tf.reduce_all(level >= upper)
        elif self.convergence is LevelTestConvergence.ANY_LE:
            converged = tf.reduce_any(level >= upper) | tf.reduce_all(level <= lower)
        else:
            raise ValueError

        return (
            converged
            | (size >= self.size_limit)
            | (step >= self.step_limit)
            | (tf.timestamp() >= cutoff_time)
        )

    def _get_next_size(self, step: int | float, size: int, elapsed_time: float) -> int:
        """Returns the next sample size while attempting to obey constraints."""
        next_size = min(
            self.size_schedule(step),  # requested value
            self.size_limit,  # sample size constraint
            elapsed_time/size * self.time_limit,  # runtime constraint
        )
        return int(next_size)


class BernoulliLevelTest(LevelTest):
    """
    Performs a sequence of statistical tests to determine whether the expected value
    of a Bernoulli random variable is above or below a level with high probability.
    """
    def __call__(
        self,
        sampler: Callable[[int], tf.Tensor],
        level: float | tf.Tensor,
        risk: float | Callable[[int], float],
        axis: int,
    ) -> LevelTestResult:
        # Record start and cutoff times
        start_time = tf.timestamp()
        cutoff_time = start_time + self.time_limit

        # Prepare for tests
        dtype = _get_dtype(level)
        step = tf.cast(0, dtype)
        size = 0
        successes = 0
        should_stop = False
        risk_schedule = (
            risk if isinstance(risk, Callable) else _get_default_risk_schedule(risk)
        )

        # Iterate until convergence or resources limits
        while not should_stop:
            step += 1
            risk = risk_schedule(step)
            next_size = self._get_next_size(step, size, tf.timestamp() - start_time)
            while size < next_size and tf.timestamp() < cutoff_time:
                batch_size = min(self.batch_limit, next_size - size)
                batch = sampler(batch_size)
                size += batch_size
                successes += tf.math.count_nonzero(batch, axis=axis, keepdims=True)
            ci = self.get_interval(risk=risk, size=size, successes=successes)
            should_stop = self.check_stopping(step, size, level, *ci, cutoff_time)

        estimate = tf.cast(successes, dtype) / tf.cast(size, dtype)
        return LevelTestResult(estimate, confidence_interval=ci, sample_size=size)

    @abstractmethod
    def get_interval(
        self,
        risk: float | tf.Tensor,
        size: int | tf.Tensor,
        successes: int | tf.Tensor
    ) -> ConfidenceInterval:
        """
        Constructs a `1 - risk` confidence interval based on the number of successes
        `k` in `n` draws from a Binomial distribution.

        Args:
            risk: Upper bound on the probability that the interval does not contain
                the true value. Same as one minus the confidence level.
            size: The number of samples drawn.
            successes: The number of successes observed.

        Returns: A ConfidenceInterval.
        """


class BayesianBernoulliLevelTest(BernoulliLevelTest):
    """Bernoulli level test using a conjugate prior for the probability of success."""
    def __init__(
        self,
        size_schedule: Callable[[int], int] | None = None,
        size_limit: int = _MAXINT,
        step_limit: int | None = None,
        time_limit: float = float("inf"),
        batch_limit: int = _MAXINT,
        convergence: LevelTestConvergence = LevelTestConvergence.ALL,
        prior: Beta | None = None,
    ):
        if prior is None:
            prior = Beta(tf.ones((), default_float()), tf.ones((), default_float()))
        elif not isinstance(prior, Beta):
            raise NotImplementedError

        super().__init__(
            size_schedule=size_schedule,
            size_limit=size_limit,
            step_limit=step_limit,
            time_limit=time_limit,
            batch_limit=batch_limit,
            convergence=convergence,
        )
        self.prior = prior

    def get_interval(
        self,
        risk: float | tf.Tensor,
        size: int | tf.Tensor,
        successes: int | tf.Tensor
    ) -> ConfidenceInterval:
        dtype = _get_dtype(risk)
        _0 = tf.zeros((), dtype=dtype)
        _1 = tf.ones((), dtype=dtype)
        a = tf.cast(self.prior.concentration0, dtype)  # alpha parameter
        b = tf.cast(self.prior.concentration1, dtype)  # beta parameter
        c = tf.cast(0.5 * risk, dtype)

        k = tf.cast(successes, dtype)
        n = tf.cast(size, dtype)
        p = Beta(a + k, n - k + b)  # posterior distribution of Binomial parameter

        lower = find_root_chandrupatla(lambda x: p.cdf(x) - c, _0, _1)[0]
        upper = find_root_chandrupatla(lambda x: p.survival_function(x) - c, _0, _1)[0]
        return ConfidenceInterval(lower, upper)


class ClopperPearsonLevelTest(BernoulliLevelTest):
    """Bernoulli level test using two-sided Clopper-Pearson confidence intervals."""

    def get_interval(
        self,
        risk: float | tf.Tensor,
        size: int | tf.Tensor,
        successes: int | tf.Tensor
    ) -> ConfidenceInterval:
        dtype = _get_dtype(risk)
        _0 = tf.zeros((), dtype=dtype)
        _1 = tf.ones((), dtype=dtype)
        n = tf.cast(size, dtype)
        k = tf.cast(successes, dtype)

        c = tf.cast(0.5 * risk, dtype)
        l = find_root_chandrupatla(lambda p: _1 - _bdtr(k - 1, n, p) - c, _0, _1)[0]
        u = find_root_chandrupatla(lambda p: _bdtr(k, n, p) - c, _0, _1,)[0]
        return ConfidenceInterval(tf.where(k != 0, l, _0), tf.where(k != n, u, _1))


class EmpiricalBernsteinLevelTest(LevelTest):
    def __init__(
        self,
        sample_range: tuple[float, float],
        size_limit: int = 2 ** 31 - 1,
        time_limit: float = float("inf"),
        batch_limit: int = 2 ** 31 - 1,
        convergence: LevelTestConvergence = LevelTestConvergence.ALL,
        size_schedule: Callable[[int], int] | None = None,
    ):
        super().__init__(
            size_schedule=size_schedule,
            batch_limit=batch_limit,
            size_limit=size_limit,
            time_limit=time_limit,
            convergence=convergence,
        )
        self.sample_range = sample_range

    def __call__(
        self,
        sampler: Callable[[int], tf.Tensor],
        level: float | tf.Tensor,
        risk: float | Callable[[int], float],
        axis: int,
    ) -> LevelTestResult:
        # Record start and cutoff times
        start_time = tf.timestamp()
        cutoff_time = start_time + self.time_limit

        # Prepare for tests
        dtype = _get_dtype(level)
        step = tf.cast(0, dtype)
        size = 0
        mean = 0
        err2 = 0  # same as `variance * size`
        should_stop = False
        risk_schedule = (
            risk if isinstance(risk, Callable) else _get_default_risk_schedule(risk)
        )

        # Iterate until convergence or resources limits
        while not should_stop:
            step += 1
            risk = risk_schedule(step)
            next_size = self._get_next_size(step, size, tf.timestamp() - start_time)
            while size < next_size and tf.timestamp() < cutoff_time:
                batch_size = min(self.batch_limit, next_size - size)
                batch = tf.cast(sampler(batch_size), dtype)
                size += batch_size
                diff = batch - mean
                mean += tf.reduce_sum(diff, axis=axis, keepdims=True) / size
                err2 += tf.reduce_sum(diff * (batch - mean), axis=axis, keepdims=True)

            ci = self.get_interval(risk, size, mean, err2)
            should_stop = self.check_stopping(step, size, level, *ci, cutoff_time)

        return LevelTestResult(mean, confidence_interval=ci, sample_size=size)

    def get_interval(
        self,
        risk: float | tf.Tensor,
        size: int | tf.Tensor,
        mean: float | tf.Tensor,
        err2: float | tf.Tensor,
    ) -> ConfidenceInterval:
        """Constructs an empirical Bernstein confidence interval."""
        dtype = _get_dtype(risk)
        isize = 1 / tf.cast(size, dtype)
        const = tf.math.log(3 / risk)
        width = self.sample_range[1] - self.sample_range[0]
        bound = isize * (tf.sqrt(2 / const * err2) + 3 * width * const)
        return ConfidenceInterval(
            lower=tf.maximum(mean - bound, self.sample_range[0]),
            upper=tf.minimum(mean + bound, self.sample_range[1]),
        )
