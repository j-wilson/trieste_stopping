from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Parameter, PriorOn, TensorData
from trieste.data import Dataset


class EmpiricalBayesParameter(Parameter):
    def __init__(
        self,
        *args: Any,
        is_transformed: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.is_transformed = is_transformed

    @abstractmethod
    def update_prior(self, dataset: Dataset) -> None:
        pass

    def log_prior_density(self) -> tf.Tensor:
        """When a prior is placed on an `unconstrained_variable` that acts as the
        parameter itself (as opposed to the transformed value thereof), we need to avoid
        differentiating through the transform when computing the log prior density."""
        if self.prior is None:
            return tf.convert_to_tensor(0.0, dtype=self.dtype)

        if self.prior_on != PriorOn.UNCONSTRAINED:
            # evaluation is in same space as prior
            return tf.reduce_sum(self.prior.log_prob(self))

        # prior on unconstrained, but evaluating log-prior in constrained space
        x = self.unconstrained_variable
        log_p = tf.reduce_sum(self.prior.log_prob(x))

        if not self.is_transformed and self.transform is not None:
            # need to include log|Jacobian| to account for coordinate transform
            log_det_jacobian = self.transform.inverse_log_det_jacobian(y, y.shape.ndims)
            log_p += tf.reduce_sum(log_det_jacobian)

        return log_p


class UniformEmpiricalBayesParameter(EmpiricalBayesParameter):
    prior: tfp.distributions.Uniform

    def __init__(
        self,
        value: TensorData,
        prior: tfp.distributions.Uniform,
        get_range: Callable[[Dataset], tuple[float, float]],
        **kwargs: Any,
    ) -> None:
        if not isinstance(prior, tfp.distributions.Uniform):
            raise ValueError

        super().__init__(value=value, prior=prior, **kwargs)
        self.get_range = get_range

    def update_prior(self, dataset: Dataset, margin: float = 1e-6) -> None:
        if not isinstance(self.prior, tfp.distributions.Uniform):
            raise TypeError

        low, high = self.get_range(dataset)

        # Clip current value to interior of updated range to avoid infinities
        width = high - low
        value = tf.clip_by_value(self, low + margin * width, high - margin * width)

        # Update prior and/or transform
        if self.prior_on is PriorOn.UNCONSTRAINED:
            low = self.transform.inverse(low)
            high = self.transform.inverse(high)

        self.prior.low.assign(low)
        self.prior.high.assign(high)
        self.assign(value)
