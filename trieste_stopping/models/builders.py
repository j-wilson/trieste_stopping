from __future__ import annotations

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Parameter, PriorOn
from gpflow.config import default_float
from gpflow.kernels import Matern52
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Constant
from gpflow.models import GPR
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer
from trieste.objectives import ObjectiveTestProblem
from trieste.space import Box
from trieste_stopping.models.models import GeneralizedGaussianProcessRegression
from trieste_stopping.models.optimizers import (
    CMAOptimizer,
    EmpiricalBayesOptimizer,
    OptimizerPipeline,
    ScipyOptimizer,
)
from trieste_stopping.models.parameters import UniformEmpiricalBayesParameter
from trieste_stopping.utils import Setting

# Default settings for (empirical Bayes) hyperpriors
empirical_variance_floor: Setting[float] = Setting(1e-6)

mean_percentile_range: Setting[tuple[float, float]] = Setting((5.0, 95.0))

kernel_variance_init: Setting[float] = Setting(1.0)
kernel_variance_range: Setting[tuple[float, float]] = Setting((0.1, 10.0))
kernel_lengthscale_median: Setting[float] = Setting(0.5)

likelihood_variance_init: Setting[float] = Setting(0.1)
likelihood_variance_range: Setting[tuple[float, float]] = Setting((1e-9, 10.0))


def build_model(
    space: ObjectiveTestProblem,
    dataset: Dataset,
    link_function: tfp.bijectors.Bijector | None = None,
    optimizer: Optimizer | None = None,
    **kwargs: Any,
) -> GaussianProcessRegression:
    """
    Builds a GaussianProcessRegression model bridge.
    """
    if optimizer is None:
        # Update hyperpriors, initialize with CMA-ES, then fine-tune with L-BFGS-B
        optimizer = OptimizerPipeline(
            optimizers=[
                EmpiricalBayesOptimizer(),
                CMAOptimizer(minimize_args={"tolfun": 1.0}),
                ScipyOptimizer()
            ]
        )

    internal_dataset = (
        dataset
        if link_function is None
        else Dataset(dataset.query_points, link_function(dataset.observations))
    )

    model = GPR(  # same as GPR but transforms y-values w/a link function `g`
        data=internal_dataset.astuple(),
        kernel=build_covariance_function(space, internal_dataset),
        likelihood=build_likelihood_function(space, internal_dataset),
        mean_function=build_mean_function(space, internal_dataset),
    )
    model_bridge = GeneralizedGaussianProcessRegression(
        model=model, optimizer=optimizer, link_function=link_function, **kwargs
    )
    return model_bridge


def build_mean_function(space: Box, dataset: Dataset) -> Constant:
    def get_range(dataset) -> tuple[float, float]:
        return tfp.stats.percentile(dataset.observations, mean_percentile_range())

    # Build mean function
    mean_function = Constant()

    # Build mean parameter
    value = tfp.stats.percentile(dataset.observations, 50)
    low, high = get_range(dataset)
    if value == low or value == high:
        value = 0.5 * (low + high)  # edge case

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
