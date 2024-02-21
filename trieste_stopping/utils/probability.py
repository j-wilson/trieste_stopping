from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, NamedTuple

import tensorflow as tf
from multipledispatch import Dispatcher
from tensorflow_probability.python import distributions
from trieste.types import TensorType

GetDistributionSupport = Dispatcher("get_distribution_support")
SupportedBaseDistributions = (
    distributions.Beta,
    distributions.Cauchy,
    distributions.Chi,
    distributions.Chi2,
    distributions.ExpGamma,
    distributions.Exponential,
    distributions.Gamma,
    distributions.Gumbel,
    distributions.Laplace,
    distributions.Logistic,
    distributions.Normal,
    distributions.StudentT,
    distributions.Uniform,
    distributions.Weibull,
)


def get_distribution_support(distribution: distributions.Distribution) -> tf.Tensor:
    return GetDistributionSupport(distribution)


@GetDistributionSupport.register(distributions.TransformedDistribution)
def _(distribution):
    return distribution.bijector(get_distribution_support(distribution.distribution))


@GetDistributionSupport.register(SupportedBaseDistributions)
def _(distribution):
    shape = distribution.batch_shape.concatenate(distribution.event_shape)
    dtype = distribution.dtype
    probs = tf.stack([tf.zeros(shape, dtype), tf.ones(shape, dtype)])
    return distribution.quantile(probs)


class SampleMeanAndVariance(NamedTuple):
    mean: TensorType
    variance: TensorType
    popsize: TensorType


def welford_estimator(
    sampler: Callable[[int], tf.Tensor],
    num_samples: int | TensorType,
    axis: int = -1,
    estimate: SampleMeanAndVariance | None = None,
    time_limit: float | None = None,
    batch_limit: int = 2 * 31 - 1,
    min_variance: float = 0.0,
) -> SampleMeanAndVariance:
    """Batch variant of Welford's algorithm for estimating means and variances."""

    # Initialize
    cutoff_time = tf.timestamp() + (float("inf") if time_limit is None else time_limit)
    if estimate is None:  # evaluate first batch
        batch = sampler(tf.cast(tf.minimum(num_samples, batch_limit), tf.int64))
        mean = tf.reduce_mean(batch, axis=axis, keepdims=True)
        variance = tf.math.reduce_variance(batch, axis=axis, keepdims=True)
        popsize = tf.cast(tf.shape(batch)[axis], mean.dtype)
        num_samples = tf.cast(num_samples, mean.dtype)
    else:  # unpack existing estimate
        mean = estimate.mean
        variance = tf.maximum(estimate.variance, min_variance)
        popsize = tf.cast(estimate.popsize, mean.dtype)
        num_samples = tf.cast(num_samples, mean.dtype)

    # Estimation loop
    def cond(mean, squared_error, popsize):
        return (popsize < num_samples) & (tf.timestamp() < cutoff_time)

    def body(mean, squared_error, popsize):
        if batch_limit is None:
            batch_size = num_samples - popsize
        else:
            batch_size = tf.minimum(num_samples - popsize, batch_limit)
        batch = sampler(tf.cast(batch_size, tf.int64))
        diffs = batch - mean

        # Note that `batch` is not guaranteed to contain `batch_size` samples
        popsize += tf.cast(tf.shape(batch)[axis], popsize.dtype)
        mean += tf.reduce_sum(diffs, axis=axis, keepdims=True) / popsize
        squared_error += tf.reduce_sum(diffs * (batch - mean), axis=axis, keepdims=True)
        return mean, squared_error, popsize

    # Iteratively update moments
    mean, squared_error, popsize = tf.while_loop(
        cond=cond, body=body, loop_vars=(mean, popsize * variance, popsize),
    )
    variance = tf.maximum(squared_error / popsize, min_variance)
    return SampleMeanAndVariance(mean=mean, variance=variance, popsize=popsize)


class EstimatorStoppingCondition(str, Enum):
    """Controls the adaptive estimator's stopping behavior"""
    ALL = "all"  # stop when all tests converge
    ANY = "any"  # stop when any test converges
    # Stop when all tests converge OR...
    ANY_GE = "any_ge"  # any test converges to a value greater-equal to the threshold
    ANY_LE = "any_le"  # any test converges to a value less-equal to the threshold


@dataclass
class AdaptiveEstimatorConfig:
    time_limit: float = float("inf")
    batch_limit: int = 2 ** 31 - 1
    popsize_initial: int = 16
    popsize_limit: int = 2 ** 31 - 1
    popsize_multiplier: float = 1.5
    stopping_condition: EstimatorStoppingCondition = EstimatorStoppingCondition.ALL


def adaptive_empirical_bernstein_estimator(
    sampler: Callable[[int], TensorType],
    sample_width: float | TensorType,
    threshold: float | TensorType,
    risk_tolerance: float | TensorType | Callable[[int], float],
    axis: int = -1,
    time_limit: float = float("inf"),
    batch_limit: int = 2 ** 31 - 1,
    popsize_limit: int = 2 ** 31 - 1,
    popsize_initial: int = 32,
    popsize_multiplier: float = 1.5,
    stopping_condition: EstimatorStoppingCondition = EstimatorStoppingCondition.ALL,
) -> tuple[SampleMeanAndVariance, tf.Tensor]:
    if popsize_initial > popsize_limit:
        raise ValueError

    risk_tolerance_schedule = (
        risk_tolerance
        if isinstance(risk_tolerance, Callable)
        else build_power_schedule(risk_tolerance)
    )
    start_time = tf.timestamp()
    cutoff_time = start_time + time_limit

    def cond(step, mean, variance, popsize, bound):
        cont = (popsize < popsize_limit) & (tf.timestamp() < cutoff_time)
        converged = tf.abs(mean - threshold) >= bound
        if stopping_condition is EstimatorStoppingCondition.ANY:
            # Continue until any test converges
            cont &= tf.reduce_any(converged)
        else:
            # Continue until all tests converge
            cont &= ~tf.reduce_all(converged)
            if stopping_condition is EstimatorStoppingCondition.ANY_GE:
                # ... or any test converges to a value greater-equal to `threshold`
                cont &= ~tf.reduce_any(converged & (mean >= threshold))
            elif stopping_condition is EstimatorStoppingCondition.ANY_LE:
                # ... or any test converges to a value less-equal to `threshold`
                cont &= ~tf.reduce_any(converged & (mean <= threshold))
        return cont

    def body(step, mean, variance, popsize, bound):
        # Generate next round of samples to update mean and variance estimates
        if time_limit is None:
            multiplier = popsize_multiplier
        else:
            elapsed_time = tf.timestamp() - start_time
            multiplier = tf.minimum(time_limit / elapsed_time, popsize_multiplier)

        mean, variance, popsize = welford_estimator(
            sampler=sampler,
            num_samples=tf.minimum(tf.math.ceil(multiplier * popsize), popsize_limit),
            estimate=SampleMeanAndVariance(mean, variance, popsize),
            batch_limit=batch_limit,
            axis=axis,
        )

        # Bound deviation of population average from true mean with high probability
        const = tf.math.log(3 / risk_tolerance_schedule(step))
        bound = (
            tf.sqrt(2 / popsize * const * variance) + 3 * sample_width * const / popsize
        )
        return step + 1, mean, variance, popsize, bound

    # Initialize by evaluating the first step
    mean, variance, popsize = welford_estimator(
        sampler=sampler,
        num_samples=popsize_initial,
        time_limit=time_limit,
        batch_limit=batch_limit,
        axis=axis,
    )
    step = tf.ones([], dtype=mean.dtype)
    const = tf.math.log(3 / risk_tolerance_schedule(step))
    bound = tf.sqrt(2 / popsize * const * variance) + 3 * sample_width * const / popsize

    # Run the loop until convergence
    step, mean, variance, popsize, bound = tf.while_loop(
        cond=cond, body=body, loop_vars=(step + 1, mean, variance, popsize, bound),
    )

    # Return PopulationMoments and deviation bound
    moments = SampleMeanAndVariance(
        mean=tf.squeeze(mean, axis=axis),
        variance=tf.squeeze(variance, axis=axis),
        popsize=popsize,
    )
    return moments, tf.squeeze(bound, axis=axis)


def build_power_schedule(total: float, p: float = 1.1) -> Callable[[int], float]:
    def schedule_func(step: int | TensorType) -> TensorType:
        return total * (p - 1) / p * (step ** -p)
    return schedule_func


def build_constant_schedule(total: float, step_limit: int) -> Callable[[int], float]:
    def schedule_func(step: int | TensorType) -> TensorType:
        if step > step_limit:
            raise ValueError
        return total / step_limit
    return schedule_func
