from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf
from gpflow.base import Module, PriorOn
from gpflow.optimizers import Scipy
from trieste.data import Dataset
from trieste.models.optimizer import create_loss_function, LossClosure, Optimizer, OptimizeResult
from trieste.space import Box
from trieste_stopping.models.parameters import EmpiricalBayesParameter
from trieste_stopping.models.utils import get_parameter_bounds
from trieste_stopping.utils.optimization import run_cmaes

NoneType = type(None)


class OptimizerPipeline(Optimizer):
    optimizer: None = None  # here to satisfy trieste API
    minimize_args: None = None
    compile: None = None

    def __init__(self, optimizers: list[Optimizer]):
        self.optimizers = optimizers

    def optimize(self, model: Module, dataset: Dataset) -> OptimizeResult | None:
        for optimizer in self.optimizers:
            result = optimizer.optimize(model, dataset)

        return result


@dataclass
class EmpiricalBayesOptimizer(Optimizer):
    optimizer: None = None
    minimize_args: None = None
    compile: None = None

    def optimize(self, model: Module, dataset: Dataset) -> None:
        for parameter in model.parameters:
            if isinstance(parameter, EmpiricalBayesParameter):
                parameter.update_prior(dataset)


@dataclass
class ScipyOptimizer(Optimizer):
    optimizer: Any = field(default_factory=Scipy)
    minimize_args: dict[str, Any] = field(default_factory=lambda: {})
    compile: bool = True

    def create_loss(self, model: Module, dataset: Dataset) -> LossClosure:
        """
        Build a loss function for the specified `model` with the `dataset` using a
        :func:`create_loss_function`.

        :param model: The model to build a loss function for.
        :param dataset: The data with which to build the loss function.
        :return: The loss function.
        """
        x = tf.convert_to_tensor(dataset.query_points)
        y = tf.convert_to_tensor(dataset.observations)
        return create_loss_function(model, (x, y), self.compile)

    def optimize(self, model: Module, dataset: Dataset) -> OptimizeResult:
        # TODO: Check for edge-cases. For example, if Scipy filters out unused
        #  variables then `bounds` may not be indexed correctly...
        loss_fn = self.create_loss(model, dataset)
        bounds = []
        variables = []
        for param in model.trainable_parameters:
            bounds.append(
                tf.reshape(get_parameter_bounds(param, unconstrained=True), (2, -1))
            )
            variables.append(param.unconstrained_variable)

        return self.optimizer.minimize(
            closure=loss_fn,
            variables=variables,
            bounds=tf.transpose(tf.concat(bounds, axis=-1)),  # n x 2
            **self.minimize_args,
        )


@dataclass
class CMAOptimizer(Optimizer):
    optimizer: None = None
    minimize_args: dict[str, Any] = field(default_factory=lambda: {})
    compile: bool = True

    def create_loss(self, model: Module, dataset: Dataset) -> LossClosure:
        """
        Build a loss function for the specified `model` with the `dataset` using a
        :func:`create_loss_function`.

        :param model: The model to build a loss function for.
        :param dataset: The data with which to build the loss function.
        :return: The loss function.
        """
        x = tf.convert_to_tensor(dataset.query_points)
        y = tf.convert_to_tensor(dataset.observations)
        return create_loss_function(model, (x, y), self.compile)

    def optimize(self, model: Module, dataset: Dataset) -> None:
        """
        Uses CMA-ES to optimize model parameters on the unit hypercube by pushing
        values forward through the quantile function of each parameter's corresponding
        prior distribution.
        """
        # Build loss function, check parameters, and create buffer
        loss_fn = self.create_loss(model, dataset)
        parameters = tuple(model.trainable_parameters)
        num_params = 0
        for param in parameters:
            if param.prior is None:
                raise NotImplementedError
            num_params += tf.size(param)
        nan_value = tf.cast(1e32, param.dtype)

        # Define loss function
        def assign(vector) -> None:
            start = 0
            for param in parameters:
                size = tf.size(param)
                vals = tf.reshape(vector[start: start + size], tf.shape(param))
                if param.prior_on is PriorOn.UNCONSTRAINED:
                    param.unconstrained_variable.assign(param.prior.quantile(vals))
                else:
                    param.assign(param.prior.quantile(vals))
                start += size

        def func(arr: np.ndarray | tf.Tensor) -> tf.Tensor:
            assign(arr)
            loss = loss_fn()
            return tf.where(tf.math.is_nan(loss), tf.cast(nan_value, loss.dtype), loss)

        # Run CMA-ES
        values, loss = run_cmaes(
            func=func,
            space=Box(lower=num_params * [0.0], upper=num_params * [1.0]),
            topk=1,
            minimize=True,
            batch_eval=False,
            compile=self.compile,
            **self.minimize_args,
        )

        # Assign best parameter values
        assign(tf.squeeze(values))


class NullOptimizer:
    """An optimizer that does nothing."""
    optimizer = None

    def optimize(self, model, dataset) -> None:
        pass