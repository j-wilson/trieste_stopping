from __future__ import annotations

import math
from functools import partial
from itertools import chain
from typing import Any, Iterable, NamedTuple

import tensorflow as tf
from numpy.polynomial.hermite import hermgauss
from trieste.acquisition.interface import (
    AcquisitionFunction,
    SingleModelAcquisitionBuilder,
    ProbabilisticModelType
)
from trieste.data import Dataset
from trieste.space import Box
from trieste.types import TensorType
from trieste_stopping.utils import run_multistart_gradient_ascent


def maximize_acquisition(
    space: Box,
    acquisition_function: AcquisitionFunction,
    custom_batches: Iterable[TensorType] | None = None,
    include_queried_points: bool = False,
    **kwargs: Any,
) -> tf.Tensor:
    """Multi-start gradient ascent routine for maximizing acquisition functions."""

    def wrapper(points: TensorType) -> tf.Tensor:
        if tf.rank(points) == 2:
            points = tf.expand_dims(points, axis=-2)

        values = acquisition_function(points)
        return tf.squeeze(values, axis=-1)   # batch_shape

    if include_queried_points:  # incl. queried points as potential starting positions
        queried_points = acquisition_function._model.get_internal_data().query_points
        custom_batches = (
            [queried_points]
            if custom_batches is None
            else chain(custom_batches, [queried_points])
        )

    best_point, best_value = run_multistart_gradient_ascent(
        fun=wrapper,
        space=space,
        custom_batches=custom_batches,
        **kwargs
    )
    return tf.experimental.numpy.atleast_2d(best_point)


class InSampleKnowledgeGradient(SingleModelAcquisitionBuilder):
    def __init__(self, num_fantasies: int = 64):
        super().__init__()
        self.num_fantasies = num_fantasies

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModelType,
        dataset: Dataset | None = None,
        num_fantasies: int | None = None,
        incumbent_index: TensorType | None = None,
        candidate_indices: TensorType | None = None,
        **kwargs: Any,
    ) -> insample_knowledge_gradient:
        if num_fantasies is None:
            num_fantasies = self.num_fantasies

        return insample_knowledge_gradient(
            model=model,
            num_fantasies=num_fantasies,
            incumbent_index=incumbent_index,
            candidate_indices=candidate_indices,
            **kwargs,
        )

    def update_acquisition_function(
        self,
        function: insample_knowledge_gradient,
        model: ProbabilisticModelType,
        dataset: Dataset | None = None,
        incumbent_index: TensorType | None = None,
        candidate_indices: TensorType | None = None,
        **kwargs: Any,
    ) -> insample_knowledge_gradient:
        function.update(incumbent_index, candidate_indices, **kwargs)
        return function


class InSampleKnowledgeGradientCache(NamedTuple):
    L: TensorType  # Cholesky factor of prior predictive covariance Cov[y(X), y(X)]
    iL_cov: TensorType  # L^{-1} Cov[f(X), f(X)]
    iL_err: TensorType  # L^{-1} y(X)
    posterior_means: TensorType  # posterior means
    incumbent_index: TensorType  # index of best point, defaults posterior minimizer
    candidate_indices: TensorType  # indices of active subset, defaults to all points


class insample_knowledge_gradient:
    def __init__(
        self,
        model,
        incumbent_index: TensorType | None = None,
        candidate_indices: tf.Tensor | None = None,
        num_fantasies: int = 64,
    ):
        self._model = model
        self._num_fantasies = num_fantasies
        self._cache = self.prepare_cache(incumbent_index, candidate_indices)

        # Generate quadrature rule
        z, dz = (
            tf.convert_to_tensor(arr, dtype=self._cache.L.dtype)
            for arr in hermgauss(deg=self._num_fantasies)
        )
        self._quadrature_z = math.sqrt(2) * z
        self._quadrature_w = (math.pi ** -0.5) * dz

    @tf.function
    def __call__(self, x: TensorType):
        # Unpack cached terms
        L11, iL11_K11, iL11_err1, mf1_given_y1, incumbent, candidates = self._cache

        # Compute moments of f(x2) | Y1
        x2 = x  # local alias to help with naming convention
        model = self._model.model
        X1, _ = model.data
        K12 = tf.linalg.adjoint(model.kernel(x2, X1))  # batch x 1 x num_train
        iL11_K12 = tf.linalg.triangular_solve(L11, K12, lower=True)
        mf2_given_y1 = (
            model.mean_function(x2) + tf.matmul(iL11_K12, iL11_err1, transpose_a=True)
        )
        vf2_given_y1 = (
            model.kernel(x2) - tf.reduce_sum(tf.square(iL11_K12), -2, keepdims=True)
        )
        vy2_given_Y1 = tf.maximum(vf2_given_y1 + model.likelihood.variance, 1e-12)

        # Prune inactive points
        K12 = tf.gather(K12, candidates, axis=-2)
        iL11_K11 = tf.gather(iL11_K11, candidates, axis=-1)
        mf1_given_y1 = tf.gather(mf1_given_y1, candidates, axis=-2)

        # Simulate Y2 | Y1, then compute E[f(X12) | Y12]
        scaled_err2 = tf.math.rsqrt(vy2_given_Y1) * self._quadrature_z
        K12_given_y1 = K12 - tf.matmul(iL11_K11, iL11_K12, transpose_a=True)
        mf1_given_y12 = mf1_given_y1 + K12_given_y1 * scaled_err2
        mf2_given_y12 = mf2_given_y1 + vf2_given_y1 * scaled_err2

        # Extract draws of min E[f(X12) | Y12]
        emin_given_y12 = tf.minimum(  # batch x 1 x num_fantasies
            mf2_given_y12, tf.reduce_min(mf1_given_y12, axis=-2, keepdims=True)
        )
        emin_given_y12 -= tf.gather(mf1_given_y12, [incumbent], axis=-2)

        # Integrate out Y2 to approximate E_{Y2}[min E[f(X12) | Y12]]
        emin_given_y1 = tf.linalg.matvec(emin_given_y12, self._quadrature_w)
        return -emin_given_y1  # sign flip

    def prepare_cache(
        self,
        incumbent_index: TensorType | None = None,
        candidate_indices: TensorType | None = None,
        out: InSampleKnowledgeGradientCache | None = None
    ) -> InSampleKnowledgeGradientCache:

        model = self._model.model
        X, Y = model.data
        residuals = Y - model.mean_function(X)
        covariance = model.kernel(X)
        predictive_covariance = tf.linalg.set_diag(
            covariance,
            tf.linalg.diag_part(covariance) + model.likelihood.variance[..., None]
        )

        L = tf.linalg.cholesky(predictive_covariance)
        iL_cov = tf.linalg.triangular_solve(L, covariance, lower=True)
        iL_err = tf.linalg.triangular_solve(L, residuals, lower=True)
        posterior_means = (
            model.mean_function(X) + tf.matmul(iL_cov, iL_err, transpose_a=True)
        )
        if incumbent_index is None:
            incumbent_index = tf.argmin(tf.squeeze(posterior_means, axis=-1), axis=0)

        if candidate_indices is None:
            candidate_indices = tf.range(tf.shape(X)[-2])

        if out:
            out.L.assign(L)
            out.iL_cov.assign(iL_cov)
            out.iL_err.assign(iL_err)
            out.posterior_means.assign(posterior_means)
            out.incumbent_index.assign(incumbent_index)
            out.candidate_indices.assign(candidate_indices)
            return out

        as_variable = partial(tf.Variable, trainable=False, shape=tf.TensorShape(None))
        return InSampleKnowledgeGradientCache(
            L=as_variable(L),
            iL_cov=as_variable(iL_cov),
            iL_err=as_variable(iL_err),
            posterior_means=as_variable(posterior_means),
            incumbent_index=as_variable(incumbent_index),
            candidate_indices=as_variable(candidate_indices)
        )

    def update(
        self,
        incumbent_index: TensorType | None = None,
        candidate_indices: TensorType | None = None
    ) -> None:
        self.prepare_cache(
            incumbent_index=incumbent_index,
            candidate_indices=candidate_indices,
            out=self._cache
        )
