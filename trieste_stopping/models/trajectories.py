from __future__ import annotations

from typing import Any, Generic, Sequence
from typing_extensions import Self

import tensorflow as tf
from gpflow.base import TensorType
from gpflow.models import GPR
from trieste_stopping.models.feature_maps import draw_kernel_feature_map, KernelMap, TFeatureMap
from trieste_stopping.models.utils import BatchModule, get_slice


class LinearTrajectory(Generic[TFeatureMap], BatchModule):
    def __init__(self, feature_map: TFeatureMap, weights: TensorType, **kwargs: Any):
        super().__init__(**kwargs)
        self.feature_map = feature_map
        self.weights = (
            weights
            if isinstance(weights, tf.Variable)
            else tf.Variable(weights, shape=tf.TensorShape(None), trainable=False)
        )

    def __call__(self, x: TensorType) -> tf.Tensor:
        return self.feature_map(x) @ self.weights

    def slice(self, index: TensorType, out: type[Self] | None = None) -> type[Self]:
        if isinstance(index, tf.Variable):
            index = index.value()

        if out is None:
            return type(self)(
                feature_map=self.feature_map[index],
                weights=get_slice(self.weights, index),
            )

        self.feature_map.slice(index, out=out.feature_map)
        out.weights.assign(get_slice(self.weights, index))
        return out

    @property
    def batch_shape(self) -> tf.Tensor:
        return tf.broadcast_dynamic_shape(
            self.feature_map.batch_shape, tf.shape(self.weights)[:-2]
        )

    @property
    def event_shape(self) -> tf.Tensor:
        return tf.shape(self.weights)[-1]


class MatheronTrajectory(BatchModule):
    def __init__(
        self,
        model: GPR,
        batch_shape: Sequence[int] | None = None,
        num_features: int | None = None,
        prior_trajectory: LinearTrajectory | None = None,
        update_trajectory: LinearTrajectory[KernelMap] | None = None,
        initialized: bool | tf.Variable = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._initialized = (
            initialized
            if isinstance(initialized, tf.Variable)
            else tf.Variable(initialized, trainable=False)
        )
        self.model = model
        self.prior_trajectory = prior_trajectory
        self.update_trajectory = update_trajectory
        if batch_shape is not None:
            self.resample(batch_shape=batch_shape, num_features=num_features)

    def __call__(self, x: TensorType) -> tf.Tensor:
        with tf.control_dependencies([tf.assert_equal(self.initialized, True)]):
            return (
                self.model.mean_function(x)
                + self.prior_trajectory(x)
                + self.update_trajectory(x)
            )

    def slice(self, index: TensorType, out: type[Self] | None = None) -> type[Self]:
        if not self.initialized:
            raise RuntimeError

        if isinstance(index, tf.Variable):
            index = index.value()

        if out is None:
            return type(self)(
                model=self.model,
                prior_trajectory=self.prior_trajectory[index],
                update_trajectory=self.update_trajectory[index],
                initialized=True,
            )

        if self.model is not out.model:
            raise NotImplementedError

        self.prior_trajectory.slice(index, out=out.prior_trajectory)
        self.update_trajectory.slice(index, out=out.update_trajectory)
        return out

    def resample(
        self, batch_shape: Sequence[int] = (), num_features: int | None = None,
    ) -> None:
        self.set_prior_trajectory(num_features=num_features, batch_shape=batch_shape)
        self.set_update_trajectory()
        self._initialized.assign(True)

    def set_prior_trajectory(
        self, batch_shape: Sequence[int] = (), num_features: int | None = None
    ) -> None:
        if not isinstance(batch_shape, tf.TensorShape):
            batch_shape = tf.TensorShape(batch_shape)

        X, Y = self.model.data
        if tf.shape(Y)[-1] != 1:
            raise NotImplementedError

        feature_map = draw_kernel_feature_map(
            kernel=self.model.kernel,
            num_inputs=tf.shape(X)[-1],
            num_features=num_features,
            batch_shape=batch_shape,
            out=(
                None
                if self.prior_trajectory is None
                else self.prior_trajectory.feature_map
            )
        )

        # Sample from standard normal
        shape = tf.concat([
            feature_map.batch_shape, feature_map.event_shape, tf.shape(Y)[-1:]], 0
        )
        weights = tf.random.normal(shape, dtype=X.dtype)
        if self.prior_trajectory is None:
            self.prior_trajectory = LinearTrajectory(
                feature_map=feature_map, weights=weights
            )
        else:
            self.prior_trajectory.weights.assign(weights)

    def set_update_trajectory(self, cholesky: TensorType | None = None) -> None:
        # Draw samples from the prior predictive
        X, Y = self.model.data
        prior_values = self.model.mean_function(X) + self.prior_trajectory(X)
        noise_variance = tf.expand_dims(self.model.likelihood.variance, -1)
        prior_predictive_values = prior_values + tf.multiply(
            tf.expand_dims(tf.sqrt(noise_variance), -1),
            tf.random.normal(shape=tf.shape(prior_values), dtype=prior_values.dtype)
        )

        # Compute Gaussian updates
        prior_predictive_errors = Y - prior_predictive_values
        if cholesky is None:
            prior_covariance = self.model.kernel(X, full_cov=True)
            prior_predictive_covariance = tf.linalg.set_diag(
                prior_covariance,
                tf.linalg.diag_part(prior_covariance) + noise_variance,
            )
            cholesky = tf.linalg.cholesky(prior_predictive_covariance)

        weights = tf.linalg.cholesky_solve(cholesky, prior_predictive_errors)
        if self.update_trajectory is None:
            self.update_trajectory = LinearTrajectory(
                weights=weights,
                feature_map=KernelMap(self.model.kernel, points=X),
            )
        else:
            self.update_trajectory.weights.assign(weights)
            self.update_trajectory.feature_map.points.assign(X)

    @property
    def batch_shape(self) -> tf.Tensor:
        return tf.broadcast_dynamic_shape(
            self.prior_trajectory.batch_shape,
            self.update_trajectory.batch_shape,
        )

    @property
    def event_shape(self) -> tf.Tensor:
        return tf.broadcast_dynamic_shape(
            self.prior_trajectory.event_shape,
            self.update_trajectory.event_shape,
        )

    @property
    def initialized(self) -> tf.Variable:
        return self._initialized
