from __future__ import annotations

from typing import Any, Sequence, TypeVar
from typing_extensions import Self

import tensorflow as tf
from gpflow import kernels
from gpflow.base import TensorType
from gpflow.config import default_float
from multipledispatch import Dispatcher
from trieste_stopping.models.utils import get_slice, BatchModule
from trieste_stopping.utils import Setting

TFeatureMap = TypeVar("TFeatureMap", bound=BatchModule)
DrawKernelFeatureMap = Dispatcher("draw_kernel_feature_map")
default_num_features: Setting[int] = Setting(default=1024)


class KernelMap(BatchModule):
    """Feature map produced by anchoring a kernel at a set of points `k(..., X)`."""
    def __init__(self, kernel, points: TensorType, **kwargs: Any):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.points = (
            points
            if isinstance(points, tf.Variable)
            else tf.Variable(points, shape=tf.TensorShape(None), trainable=False)
        )

    def __call__(self, x: TensorType) -> tf.Tensor:
        return self.kernel(x, self.points)

    def slice(self, index: TensorType, out: type[Self] | None = None) -> type[Self]:
        shared = tf.rank(self.points) < 3  # are points shared by all batches?
        points = self.points if shared else get_slice(self.points, index)
        if out is None:
            return type(self)(kernel=self.kernel, points=tf.identity(points))

        if out is self:
            if not shared:
                self.points.assign(points)
            return self

        if out.kernel is not self.kernel:
            raise NotImplementedError

        out.points.assign(points)
        return out

    @property
    def batch_shape(self) -> tf.Tensor:
        return tf.shape(self.points)[:-2]

    @property
    def event_shape(self) -> tf.Tensor:
        return tf.shape(self.points)[-2:-1]


class ComplexExponentialMap(BatchModule):
    """Feature map representing complex exponential functions $\phi(x) = \exp(iw'x)$
    as pairs of trigonometric features $sin(w'x)$ and $cos(w'x)$. We have,

    .. code-block:: text

        \Re[\phi(x1) \overline{\phi(x2)}] = \Re[e^{iw'(x1 - x2)}]
                                          = cos(w'(x1 - x2))
                                          = sin(w'x1)sin(w'x2) + cos(w'x1)cos(w'x2)

        where $\Re$ denotes the real part and $\overline$ is the complex conjugate.
    """
    def __init__(self, kernel, weights: TensorType, **kwargs: Any):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.weights = (
            weights
            if isinstance(weights, tf.Variable)
            else tf.Variable(weights, shape=tf.TensorShape(None), trainable=False)
        )

    def __call__(self, x: TensorType) -> tf.Tensor:
        proj = self.kernel.scale(x) @ self.weights
        feat = tf.concat((tf.sin(proj), tf.cos(proj)), axis=-1)
        ampl = tf.sqrt(2 / tf.reduce_prod(self.event_shape) * self.kernel.variance)
        return ampl[..., None, None] * feat  # [batch..., x.shape[-2], num_features]

    def slice(self, index: TensorType, out: type[Self] | None = None) -> type[Self]:
        if out is None:
            return type(self)(self.kernel, weights=get_slice(self.weights, index))

        if self.kernel is not out.kernel:
            raise NotImplementedError

        out.weights.assign(get_slice(self.weights, index))
        return out

    @property
    def batch_shape(self) -> tf.Tensor:
        return tf.shape(self.weights)[:-2]

    @property
    def event_shape(self) -> tf.Tensor:
        return 2 * tf.shape(self.weights)[-1:]


def draw_kernel_feature_map(
    kernel: kernels.Kernel,
    num_inputs: int | TensorType,
    num_features: int | TensorType | None = None,
    batch_shape: Sequence[int] = (),
    out: TFeatureMap | None = None,
    **kwargs: Any,
) -> TFeatureMap:
    return DrawKernelFeatureMap(
        kernel,
        num_inputs=num_inputs,
        num_features=num_features,
        batch_shape=batch_shape,
        out=out,
        **kwargs,
    )


@DrawKernelFeatureMap.register(
    (kernels.Matern12, kernels.Matern32, kernels.Matern52, kernels.SquaredExponential)
)
def _(
    kernel,
    num_inputs: int | TensorType,
    num_features: int | TensorType | None,
    batch_shape: int | TensorType,
    out: ComplexExponentialMap | None = None,
    **ignore: Any,
) -> ComplexExponentialMap:
    if out is not None:
        if out.kernel is not kernel:
            raise NotImplementedError
        if not isinstance(out, ComplexExponentialMap):
            raise ValueError

    if num_features is None:
        num_features = default_num_features()

    if num_features % 2:
        raise NotImplementedError

    if not isinstance(batch_shape, tf.TensorShape):
        batch_shape = tf.TensorShape(batch_shape)

    # Generate standard normal weights
    shape = batch_shape + (num_inputs, num_features // 2)
    weights = tf.random.normal(shape, dtype=default_float())
    if not isinstance(kernel, kernels.SquaredExponential):
        # Convert to multivariate student-t
        nu = tf.cast(
            1 / 2 if isinstance(kernel, kernels.Matern12)
            else 3 / 2 if isinstance(kernel, kernels.Matern32)
            else 5 / 2,
            dtype=default_float(),
        )
        weights *= tf.math.rsqrt(
            tf.random.gamma(
                shape=shape[:-2] + (1,) + shape[-1:],
                dtype=default_float(),
                alpha=nu,
                beta=nu,
            )
        )

    if out is None:
        return ComplexExponentialMap(kernel, weights)

    out.weights.assign(weights)
    return out
