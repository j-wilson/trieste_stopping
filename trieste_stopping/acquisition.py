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
from trieste_stopping.models import get_link_function
from trieste_stopping.utils import get_expected_value, run_multistart_gradient_ascent


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
    def __init__(self, num_samples: int = 16):
        super().__init__()
        self.num_samples = num_samples

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModelType,
        dataset: Dataset | None = None,
        num_samples: int | None = None,
        best_index: TensorType | None = None,
        candidate_indices: TensorType | None = None,
        **kwargs: Any,
    ) -> insample_knowledge_gradient:
        if num_samples is None:
            num_samples = self.num_samples

        return insample_knowledge_gradient(
            model=model,
            num_samples=num_samples,
            best_index=best_index,
            candidate_indices=candidate_indices,
            **kwargs,
        )

    def update_acquisition_function(
        self,
        function: insample_knowledge_gradient,
        model: ProbabilisticModelType,
        dataset: Dataset | None = None,
        best_index: TensorType | None = None,
        candidate_indices: TensorType | None = None,
        **kwargs: Any,
    ) -> insample_knowledge_gradient:
        function.update(best_index, candidate_indices, **kwargs)
        return function


class InSampleKnowledgeGradientCache(NamedTuple):
    L: TensorType  # Cholesky factor of prior predictive covariance Cov[y(X), y(X)]
    iL_cov: TensorType  # L^{-1} Cov[f(X), f(X)]
    iL_err: TensorType  # L^{-1} y(X)
    posterior: tuple[TensorType, TensorType]  # posterior means and variances
    best_index: TensorType  # index of best point, defaults posterior minimizer
    active_indices: TensorType  # indices of active subset, defaults to all points


class insample_knowledge_gradient:
    def __init__(
        self,
        model,
        best_index: TensorType | None = None,
        candidate_indices: tf.Tensor | None = None,
        num_samples: int = 64,
        parallel_iterations: int = 1,
    ):
        self._model = model
        self._num_samples = num_samples
        self._parallel_iterations = parallel_iterations
        self._cache = self.prepare_cache(best_index, candidate_indices)

        # Generate quadrature rule
        z, dz = (
            tf.convert_to_tensor(arr, dtype=self._cache.L.dtype)
            for arr in hermgauss(deg=self._num_samples)
        )
        self._quadrature_z = math.sqrt(2) * z  # abscissae
        self._quadrature_w = (math.pi ** -0.5) * dz  # weights

    @tf.function
    def __call__(self, x: TensorType) -> tf.Tensor:
        """
        Compute the knowledge acquisition function on the set of observed points.

        As notation, we use `1` to denote quantities related to the set of observed
        points, `2` to denote quantities related to the input `x`, and `3` to denote
        the union of `1` and `2`.
        """
        # Unpack cached terms
        L11, iL11_K11, iL11_err1, (mf1_given_y1, vf1_given_y1), best, keep = self._cache

        # Compute moments of $f(x2) | y1$ and $y(x2) | y1$
        X2 = x  # local alias to help with naming convention
        gp = self._model.model
        X1, _ = gp.data
        k12 = tf.linalg.adjoint(gp.kernel(X2, X1))  # batch x 1 x num_train
        iL11_k12 = tf.linalg.triangular_solve(L11, k12, lower=True)
        k21_iK11_err1 = tf.matmul(iL11_k12, iL11_err1, transpose_a=True)
        mf2_given_y1 = gp.mean_function(X2) + k21_iK11_err1
        k21_iK11_k12 = tf.reduce_sum(tf.square(iL11_k12), -2, keepdims=True)
        vf2_given_y1 = gp.kernel(X2) - k21_iK11_k12
        vy2_given_y1 = tf.maximum(vf2_given_y1 + gp.likelihood.variance, 1e-12)

        # Prune inactive points
        k12 = tf.gather(k12, keep, axis=-2)
        iL11_K11 = tf.gather(iL11_K11, keep, axis=-1)
        mf1_given_y1 = tf.gather(mf1_given_y1, keep, axis=-2)

        # Simulate $y2 | y1$, then compute $E[f(X3) | y3]$
        scaled_err2 = tf.math.rsqrt(vy2_given_y1) * self._quadrature_z
        k12_given_y1 = k12 - tf.matmul(iL11_K11, iL11_k12, transpose_a=True)
        mf1_given_y3 = mf1_given_y1 + k12_given_y1 * scaled_err2
        mf2_given_y3 = mf2_given_y1 + vf2_given_y1 * scaled_err2

        # Account for a link function, $E[g^{-1}(f(X3)) | y3]$
        link = get_link_function(self._model, skip_identity=True)
        if link is not None:
            vf1_given_y3 = vf1_given_y1 - tf.square(k12_given_y1) / vy2_given_y1
            vf2_given_y3 = vf2_given_y1 - tf.square(vf2_given_y1) / vy2_given_y1
            mf1_given_y3, mf2_given_y3 = (
                get_expected_value(
                    mean=m,
                    variance=v,
                    inverse=link,
                    num_samples=self._num_samples,
                    parallel_iterations=self._parallel_iterations,  # limits memory use
                )
                for m, v in ((mf1_given_y3, vf1_given_y3), (mf2_given_y3, vf2_given_y3))
            )

        # Extract draws of $min E[g^{-1}(f(X3)) | y3]$
        emin_given_y3 = tf.gather(mf1_given_y3, [best], axis=-2) - tf.minimum(
            mf2_given_y3, tf.reduce_min(mf1_given_y3, axis=-2, keepdims=True)
        )  # batch x 1 x num_samples

        # Integrate out y2 to estimate $E_{y2}[min E[g^{-1}(f(X3)) | y3]$
        emin_given_y1 = tf.linalg.matvec(emin_given_y3, self._quadrature_w)
        return emin_given_y1

    def prepare_cache(
        self,
        best_index: TensorType | None = None,
        active_indices: TensorType | None = None,
        out: InSampleKnowledgeGradientCache | None = None
    ) -> InSampleKnowledgeGradientCache:
        gp = self._model.model
        link = get_link_function(self._model)
        X, Y = gp.data

        err = Y - gp.mean_function(X)
        Kff = gp.kernel(X)
        Kyy = tf.linalg.set_diag(
            Kff,
            tf.linalg.diag_part(Kff) + gp.likelihood.variance[..., None]
        )

        L = tf.linalg.cholesky(Kyy)
        iL_cov = tf.linalg.triangular_solve(L, Kff, lower=True)
        iL_err = tf.linalg.triangular_solve(L, err, lower=True)

        means = gp.mean_function(X) + tf.matmul(iL_cov, iL_err, transpose_a=True)
        variances = tf.expand_dims(
            tf.linalg.diag_part(Kff) - tf.reduce_sum(tf.square(iL_cov), axis=-2),
            axis=-1
        )
        if best_index is None:
            expected_values = get_expected_value(means, variances, inverse=link)
            best_index = tf.argmin(tf.squeeze(expected_values, axis=-1), axis=0)

        if active_indices is None:  # can use InSampleExpectedMinimum if lots of points
            active_indices = tf.range(tf.shape(X)[-2])

        if out:
            out.L.assign(L)
            out.iL_cov.assign(iL_cov)
            out.iL_err.assign(iL_err)
            out.posterior[0].assign(means)
            out.posterior[1].assign(variances)
            out.best_index.assign(best_index)
            out.active_indices.assign(active_indices)
            return out

        as_variable = partial(tf.Variable, trainable=False, shape=tf.TensorShape(None))
        return InSampleKnowledgeGradientCache(
            L=as_variable(L),
            iL_cov=as_variable(iL_cov),
            iL_err=as_variable(iL_err),
            posterior=(as_variable(means), as_variable(variances)),
            best_index=as_variable(best_index),
            active_indices=as_variable(active_indices)
        )

    def update(
        self,
        best_index: TensorType | None = None,
        candidate_indices: TensorType | None = None
    ) -> None:
        self.prepare_cache(best_index, candidate_indices, out=self._cache)
