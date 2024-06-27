from __future__ import annotations

from math import pi
from typing import Any, Callable

import tensorflow as tf
import tensorflow_probability as tfp
from multipledispatch import Dispatcher
from numpy.polynomial.hermite import hermgauss

NoneType = type(None)
GetExpectedValue = Dispatcher("get_expected_value")
GetDistributionSupport = Dispatcher("get_distribution_support")

SupportedBaseDistributions = (
    tfp.distributions.Beta,
    tfp.distributions.Cauchy,
    tfp.distributions.Chi,
    tfp.distributions.Chi2,
    tfp.distributions.ExpGamma,
    tfp.distributions.Exponential,
    tfp.distributions.Gamma,
    tfp.distributions.Gumbel,
    tfp.distributions.Laplace,
    tfp.distributions.Logistic,
    tfp.distributions.Normal,
    tfp.distributions.StudentT,
    tfp.distributions.Uniform,
    tfp.distributions.Weibull,
)


def get_distribution_support(distribution: tfp.distributions.Distribution) -> tf.Tensor:
    return GetDistributionSupport(distribution)


@GetDistributionSupport.register(tfp.distributions.TransformedDistribution)
def _(distribution):
    return distribution.bijector(get_distribution_support(distribution.distribution))


@GetDistributionSupport.register(SupportedBaseDistributions)
def _(distribution):
    shape = distribution.batch_shape.concatenate(distribution.event_shape)
    dtype = distribution.dtype
    probs = tf.stack([tf.zeros(shape, dtype), tf.ones(shape, dtype)])
    return distribution.quantile(probs)


def get_expected_value(
    mean: tf.Tensor,
    variance: tf.Tensor,
    forward: Callable[[tf.Tensor], tf.Tensor] | tfp.bijectors.Bijector | None = None,
    inverse: tfp.bijectors.Bijector | None = None,
    **kwargs: Any,
) -> tf.Tensor:
    """
    Computes or estimates the expected value of a Gaussian random variable pushed
    forward through a function.

    Args:
        mean: The random variable's mean.
        variance: The random variable's variance.
        forward: A optional forward method.
        inverse: An optional inverse method; must be a Bijector instance or None.
        **kwargs: Keyword arguments passed to subroutines.

    Returns: The expected value of the transformed random variable.
    """
    if inverse is not None:
        if forward is not None:
            raise ValueError("Only one of `forward` or `inverse` may be passed.")

        forward = (  # extract or construct a forward method
            inverse if isinstance(inverse, tfp.bijectors.Identity)
            else inverse.bijector if isinstance(inverse, tfp.bijectors.Invert)
            else tfp.bijectors.Invert(inverse)
        )

    return GetExpectedValue(
        forward,
        mean=mean,
        variance=variance,
        inverse=inverse,
        **kwargs
    )


@GetExpectedValue.register(object)
def _(
    forward,
    mean,
    variance,
    num_samples: int = 16,
    parallel_iterations: int | None = None,
    **ignore: Any
):
    z, dz = (tf.cast(ndarray, dtype=mean.dtype) for ndarray in hermgauss(num_samples))
    if parallel_iterations is None:
        draws = forward(mean[..., None] + tf.sqrt(2 * variance[..., None]) * z)
        return (pi ** -0.5) * tf.reduce_sum(draws * dz, axis=-1)

    def fn(accum: tf.Tensor, elems: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        z, dz = elems
        return accum + forward(mean + tf.sqrt(2 * variance) * z) * dz

    shape = tf.broadcast_dynamic_shape(tf.shape(mean), tf.shape(variance))
    accum = tf.zeros(shape, dtype=mean.dtype)
    value = tf.foldl(fn, (z, dz), accum, parallel_iterations=parallel_iterations)
    return (pi ** -0.5) * value


@GetExpectedValue.register((tfp.bijectors.Identity, NoneType))
def _(forward, mean, variance,  **ignore: Any):
    return tf.identity(mean)


@GetExpectedValue.register(tfp.bijectors.NormalCDF)
def _(forward, mean, variance, **ignore: Any):
    return forward(mean * tf.math.rsqrt(variance + 1))
