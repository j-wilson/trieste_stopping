from __future__ import annotations

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Parameter, PriorOn
from gpflow.config import default_float
from gpflow.models import GPR
from gpflow.kernels import Matern52
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Constant
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer
from trieste.space import Box
from trieste_stopping.models.optimizers import (
    EmpiricalBayesOptimizer,
    OptimizerPipeline,
    ScipyOptimizer,
)
from trieste_stopping.models.parameters import UniformEmpiricalBayesParameter
from trieste_stopping.settings import (
    empirical_variance_floor,
    kernel_lengthscale_median,
    kernel_variance_init,
    kernel_variance_range,
    mean_percentile_range,
    likelihood_variance_init,
    likelihood_variance_range,
)


def build_model(
    space: Box,
    dataset: Dataset,
    optimizer: Optimizer | None = None,
    **kwargs: Any,
) -> GaussianProcessRegression:
    """Builds a GaussianProcessRegression model bridge."""
    if optimizer is None:
        optimizer = OptimizerPipeline([EmpiricalBayesOptimizer(), ScipyOptimizer()])

    model = GPR(
        data=dataset.astuple(),
        mean_function=build_mean_function(space, dataset),
        kernel=build_covariance_function(space, dataset),
        likelihood=build_likelihood_function(space, dataset)
    )
    return GaussianProcessRegression(model, optimizer=optimizer, **kwargs)


def build_mean_function(space: Box, dataset: Dataset) -> Constant:
    def get_range(dataset) -> tuple[float, float]:
        return tfp.stats.percentile(dataset.observations, mean_percentile_range())

    # Build mean function
    mean_function = Constant()

    # Build mean parameter
    value = tfp.stats.percentile(dataset.observations, 50)
    low, high = get_range(dataset)
    mean_function.c = UniformEmpiricalBayesParameter(
        value=value,
        prior=tfp.distributions.Uniform(
            low=tf.Variable(low, dtype=default_float(), trainable=False),
            high=tf.Variable(high, dtype=default_float(), trainable=False),
        ),
        get_range=get_range,
    )
    return mean_function


def build_covariance_function(space: Box, dataset: Dataset) -> Matern52:
    # Build covariance function
    kernel = Matern52()

    # Build lengthscale parameter
    median = kernel_lengthscale_median() * (space.upper - space.lower)
    lengthscale_prior = tfp.distributions.LogNormal(
        loc=tf.Variable(tf.math.log(median), dtype=default_float(), trainable=False),
        scale=tf.Variable(1, dtype=default_float(), trainable=False),
    )
    kernel.lengthscales = Parameter(value=median,  prior=lengthscale_prior)

    # Build variance parameter
    def get_range(dataset: Dataset) -> tuple[float, float]:
        variance = tf.maximum(
            tf.math.reduce_variance(dataset.observations),
            empirical_variance_floor()
        )
        return tuple(c * variance for c in kernel_variance_range())

    variance = tf.maximum(
        tf.math.reduce_variance(dataset.observations), empirical_variance_floor()
    )
    low, high = map(tf.math.log, (c * variance for c in kernel_variance_range()))
    variance_prior = tfp.distributions.Uniform(
        low=tf.Variable(low, dtype=default_float(), trainable=False),
        high=tf.Variable(high, dtype=default_float(), trainable=False),
    )
    kernel.variance = UniformEmpiricalBayesParameter(
        value=variance * kernel_variance_init(),
        prior=variance_prior,
        prior_on=PriorOn.UNCONSTRAINED,
        get_range=get_range,
        transform=tfp.bijectors.Exp(),
        is_transformed=True,  # define the parameter in log-space
    )
    return kernel


def build_likelihood_function(space: Box, dataset: Dataset) -> Gaussian:
    # Build likelihood function
    likelihood = Gaussian()

    # Build variance parameter
    def get_range(dataset: Dataset) -> tuple[float, float]:
        variance = tf.maximum(
            tf.math.reduce_variance(dataset.observations),
            empirical_variance_floor()
        )
        return tuple(c * variance for c in likelihood_variance_range())

    variance = tf.maximum(
        tf.math.reduce_variance(dataset.observations), empirical_variance_floor()
    )
    low, high = map(tf.math.log, (c * variance for c in likelihood_variance_range()))
    variance_prior = tfp.distributions.Uniform(
        low=tf.Variable(low, dtype=default_float(), trainable=False),
        high=tf.Variable(high, dtype=default_float(), trainable=False),
    )
    likelihood.variance = UniformEmpiricalBayesParameter(
        value=variance * likelihood_variance_init(),
        prior=variance_prior,
        prior_on=PriorOn.UNCONSTRAINED,
        get_range=get_range,
        transform=tfp.bijectors.Exp(),
        is_transformed=True,  # define the parameter in log-space
    )
    return likelihood
