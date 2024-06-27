from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generic, Iterator, TypeVar
from typing_extensions import Self

import tensorflow as tf
from abc import ABC, abstractmethod
from gpflow.base import _IS_PARAMETER, Module, Parameter, PriorOn
from trieste.types import TensorType
from trieste_stopping.utils import get_distribution_support

TBatchModule = TypeVar("TBatchTransform", bound="BatchTransform")


class BatchModule(ABC):
    def __init_subclass__(cls, **ignore: Any):
        cls.__getitem__ = cls.slice

    @abstractmethod
    def slice(self, index: TensorType, out: type[Self] | None = None) -> type[Self]:
        pass

    @abstractmethod
    def __call__(self, x: TensorType) -> tf.Tensor:
        pass

    @property
    @abstractmethod
    def batch_shape(self) -> tf.TensorShape:
        pass

    @property
    @abstractmethod
    def event_shape(self) -> tf.TensorShape:
        pass


def get_parameters(module: Module) -> Iterator[tuple[str, Parameter]]:
    for path, param in module._flatten(predicate=_IS_PARAMETER, with_path=True):
        yield ".".join(path), param


def set_parameters(module: Module, values: dict[str, Any]) -> None:
    for name, param in get_parameters(module):
        if name in values:
            if param.transform is None:
                param.assign(values[name])
            else:
                val = tf.convert_to_tensor(values[name], dtype_hint=param.dtype)
                unconstrained_val = param.transform.inverse(val)
                param.unconstrained_variable.assign(unconstrained_val)


def get_parameter_bounds(param: Parameter, unconstrained: bool = False) -> tf.Tensor:
    shape = tf.shape(param)
    if param.prior is None:
        inf = tf.cast(float("inf"), param.dtype)
        bounds = tf.stack([tf.fill(shape, -inf), tf.fill(shape, inf)])
        is_unconstrained = True
    else:
        bounds = get_distribution_support(param.prior)
        is_unconstrained = param.prior_on is PriorOn.UNCONSTRAINED

    # Check whether `bounds` are in the desired space
    if param.transform is not None:
        if unconstrained and not is_unconstrained:
            bounds = param.transform.inverse(bounds)
        elif is_unconstrained and not unconstrained:
            bounds = param.transform.forward(bounds)

    # Return bounds as (2, *param.shape)
    return tf.stack([tf.broadcast_to(bound, shape) for bound in bounds])


def derelativize_index(
    index: TensorType,
    source: TensorType,
    shape: tf.TensorShape | None = None,
) -> tf.Tensor:
    if source.dtype == tf.bool:
        shape = tf.shape(source) if shape is None else shape
        source = tf.where(source)
    elif index.dtype == tf.bool and shape is None:
        raise ValueError

    if index.dtype == tf.bool:
        indices = tf.boolean_mask(source, index)
        return tf.scatter_nd(
            indices=indices,
            updates=tf.fill(tf.shape(indices)[:-1], True),
            shape=tf.cast(shape, indices.dtype),
        )

    return tf.gather(source, index)


def get_slice(x: TensorType, index: TensorType) -> tf.Tensor:
    return tf.boolean_mask(x, index) if index.dtype == tf.bool else tf.gather(x, index)


class SliceManager(Generic[TBatchModule], BatchModule):
    def __init__(
        self,
        module: TBatchModule,
        index: TensorType | None = None,
        index_dtype: type = tf.int64,
        **kwargs: Any,
    ):
        if tf.rank(module.batch_shape) != 1:
            raise NotImplementedError

        super().__init__(**kwargs)
        self._module = module

        if index is None:
            index = self._get_full_index(dtype=index_dtype)

        self._index = (
            index
            if isinstance(index, tf.Variable)
            else tf.Variable(index, shape=[None], trainable=False)
        )
        self._sliced_module = self.module.slice(self.index)

    def __call__(self, x: TensorType, **kwargs: Any) -> tf.Tensor:
        return self.sliced_module(x, **kwargs)

    def slice(
        self,
        index: TensorType,
        out: type[Self] | None = None,
        relative: bool = True,
    ) -> type[Self]:
        if relative:
            index = derelativize_index(
                index=index, source=self.index, shape=self.module.batch_shape
            )

        if out is self:  # update index and write to live transform
            self.index.assign(index)
            self.module.slice(index=self.index, out=self.sliced_module)
            return self

        if out is None:
            return type(self)(module=self.module, index=index)

        self.module.slice(self._get_full_index(), out=out.transform)
        return out.slice(index=index, relative=False, out=out)

    def islice(self, index: TensorType, relative: bool = True) -> Self:
        return self.slice(index, relative=relative, out=self)

    def unslice(self) -> Self:
        return self.islice(self._get_full_index(), relative=False)

    @contextmanager
    def islice_ctx(self, index: TensorType, relative: bool = True) -> None:
        prev = tf.identity(self.index)
        try:
            self.islice(index=index, relative=relative)
            yield
        finally:
            self.islice(index=prev, relative=False)

    def _get_full_index(self, dtype: type | None = None) -> tf.Tensor:
        if tf.rank(self.module.batch_shape) != 1:
            raise NotImplementedError

        if dtype is None:
            dtype = self.index.dtype

        if dtype == tf.bool:
            return tf.fill(self.module.batch_shape, True)

        if dtype not in (tf.int32, tf.int64):
            raise ValueError

        return tf.range(self.module.batch_shape, dtype=dtype)

    @property
    def index(self) -> tf.Variable:
        return self._index

    @property
    def module(self) -> TBatchModule:
        return self._module

    @property
    def sliced_module(self) -> TBatchModule:
        return self._sliced_module

    @property
    def batch_shape(self) -> tf.TensorShape:
        return self.sliced_module.batch_shape

    @property
    def event_shape(self) -> tf.TensorShape:
        return self.sliced_module.event_shape
