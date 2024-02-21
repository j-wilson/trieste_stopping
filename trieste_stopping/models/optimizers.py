from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf
from gpflow.base import Module
from gpflow.optimizers import Scipy
from trieste.data import Dataset
from trieste.models.optimizer import create_loss_function, LossClosure, Optimizer
from trieste_stopping.models.parameters import EmpiricalBayesParameter
from trieste_stopping.models.utils import get_parameter_bounds

NoneType = type(None)


class OptimizerPipeline(Optimizer):
    optimizer: None = None  # here to satisfy trieste API
    minimize_args: dict[str, Any] = field(default_factory=lambda: {})
    compile: bool = True

    def __init__(self, optimizers: list[Optimizer]):
        self.optimizers = optimizers

    def optimize(self, model: Module, dataset: Dataset) -> None:
        for optimizer in self.optimizers:
            optimizer.optimize(model, dataset)


@dataclass
class EmpiricalBayesOptimizer(Optimizer):
    optimizer: None = None
    minimize_args: dict[str, Any] = field(default_factory=lambda: {})
    compile: bool = True

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

    def optimize(self, model: Module, dataset: Dataset) -> None:
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
