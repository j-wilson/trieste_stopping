from __future__ import annotations

from functools import partial
from itertools import chain, count, zip_longest
from sys import maxsize
from time import monotonic
from typing import Any, Callable, Iterable, Iterator, Sequence, overload
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from cmaes import CMA
from scipy.optimize import minimize, OptimizeResult
from scipy.optimize._optimize import _wrap_callback
from trieste.acquisition.optimizer import _perform_parallel_continuous_optimization
from trieste.space import Box
from trieste.types import TensorType


class StopMinimize(Exception):
    __slots__ = ("result",)

    def __init__(self, result: OptimizeResult) -> None:
        self.result = result


def minimize_with_stopping(
    fun: Callable[[np.ndarray, ...], float],
    x0: np.ndarray,
    callback: Callable[..., None] | None = None,
    stop_callback: Callable[[OptimizeResult], bool] | None = None,
    method: str | None = None,
    **kwargs,
) -> OptimizeResult:
    iteration_counter = count()
    if stop_callback is not None:
        # Wrap user callback to accept OptimizeResult as it sole argument
        user_callback = None if callback is None else _wrap_callback(callback, method)

        def callback(intermediate_result: OptimizeResult):
            next(iteration_counter)  # increment the counter
            if user_callback:
                user_callback(intermediate_result)

            if stop_callback(intermediate_result):
                raise StopMinimize(result=intermediate_result)

    try:
        return minimize(
            fun=fun,
            x0=x0,
            method=method,
            callback=callback,
            **kwargs
        )
    except StopMinimize as e:
        return OptimizeResult(
            x=e.result.x,
            fun=e.result.fun,
            nfev=-1,
            nit=next(iteration_counter),
            message="`scipy.minimize` terminated due to early stopping callback.",
            success=True,
        )


@overload
def reduce_topk(
    inputs: Iterable[TensorType],
    k: int,
    axis: int = -1,
    smallest: bool = False,
    callback: Callable[[TensorType, ...], bool] | None = None,
) -> tf.Tensor:
    pass


@overload
def reduce_topk(
    inputs: Iterable[Iterable[TensorType]],
    k: int,
    axis: int = -1,
    smallest: bool = False,
    callback: Callable[[TensorType, ...], bool] | None = None,
) -> tuple[tf.Tensor, ...]:
    pass


def reduce_topk(inputs, k, axis=-1, smallest=False, callback=None):
    step: int | None = None
    rank: tf.Tensor | None = None
    swap: Callable[[TensorType], tf.Tensor] | None = None
    join: Callable[[TensorType, TensorType], tf.Tensor] | None = None

    tensor_mode: bool | None = None
    topk_values: TensorType | None = None
    topk_extras: list[TensorType, ...] | None = None
    for step, entry in enumerate(inputs):
        # Unpack next entry
        is_tensor = tf.is_tensor(entry)
        if tensor_mode is None:
            tensor_mode = is_tensor
        elif is_tensor != tensor_mode:  # check typing
            raise TypeError("Elements of `inputs` had incompatible types.")

        if is_tensor:
            values = entry
            extras = ()
        else:
            values, *extras = entry

        # Define join and swap operations
        if step == 0:
            rank = tf.rank(values)
            axis = rank + axis if axis < 0 else axis
            if axis != rank - 1:
                swap = partial(
                    tf.experimental.numpy.swapaxes, axis1=axis, axis2=rank - 1
                )

            def join(old: TensorType, new: TensorType) -> tf.Tensor:
                return tf.concat((old, new), axis=rank - 1)

        if smallest:
            values = -values

        if swap:  # move target axis to the inside for tf.math.top_k
            values = swap(values)
            extras = map(swap, extras)

        if step:  # merge old and new terms
            values = join(topk_values, values)
            extras = (join(old, new) for old, new in zip_longest(topk_extras, extras))

        if tf.shape(values)[-1] <= k:  # ensure robustness to in-place updates
            topk_values = tf.identity(values)
            topk_extras = [tf.identity(x) for x in extras]
        else:  # extract top-k terms
            topk_values, indices = tf.math.top_k(values, k=k, sorted=False)
            topk_extras = [tf.gather(x, indices, batch_dims=rank - 1) for x in extras]

        if callback:
            topk_values_ = -topk_values if smallest else topk_values
            if swap:
                callback(swap(topk_values_), *map(swap, topk_extras))
            else:
                callback(topk_values_, *topk_extras)

    if step is None:
        raise RuntimeError("Received an empty iterator for `parts`.")

    if smallest:
        topk_values *= -1

    if swap:  # restore axis to original position
        topk_values = swap(topk_values)
        topk_extras = [swap(x) for x in topk_extras]

    return topk_values if tensor_mode else (topk_values, *topk_extras)


def run_cmaes(
    func: Callable[[TensorType], TensorType],
    space: Box,
    minimize: bool = False,
    initial_loc: TensorType | None = None,
    initial_scale: TensorType | None = None,
    population_size: int | None = None,
    seed: int | None = None,
    topk: int = 1,
    tolx: float = 1e-12,
    tolfun: float = 1e-12,
    compile: bool = True,
    time_limit: float = float("inf"),
    step_limit: int = maxsize,
    batch_eval: bool = True,
    **kwargs: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    start_time = monotonic()
    if initial_loc is None:
        initial_loc = 0.5 * (space.lower + space.upper)

    if initial_scale is None:
        initial_scale = tf.reduce_mean(space.upper - space.lower) / 3

    if seed is None:
        seed = tf.random.uniform(shape=(), maxval=2 ** 31 - 1, dtype=tf.int64)

    # Initialize the CMA minimizer
    cma = CMA(
        seed=seed,
        mean=np.asarray(initial_loc),
        sigma=float(initial_scale),
        bounds=np.stack([space.lower, space.upper], axis=-1),
        population_size=population_size,
        **kwargs,
    )
    cma._tolx = initial_scale * tolx
    cma._tolfun = tolfun

    # Construct buffers and closure
    X = tf.Variable(tf.zeros([cma.population_size, space.dimension], dtype=tf.float64))
    Y = tf.Variable(tf.zeros([cma.population_size], dtype=tf.float64))

    def closure() -> None:
        Y.assign(func(X) if batch_eval else [func(x) for x in tf.unstack(X)])

    if compile:
        closure = tf.function(closure)

    def cma_generator() -> Iterator[tuple[tf.Tensor, tf.Tensor]]:
        for _ in range(step_limit):
            # Evaluate next population
            X.assign([cma.ask() for _ in range(cma.population_size)])
            closure()

            # Pass results to the CMA minimizer
            cma.tell(list(zip(X, Y if minimize else -Y)))
            yield Y, X

            # Check stopping conditions
            if cma.should_stop() or monotonic() - start_time > time_limit:
                break

    # Run CMA-ES and accumulate best results
    topk_Y, topk_X = reduce_topk(cma_generator(), k=topk, smallest=minimize)

    # Return top-k inputs and outputs
    return topk_X, topk_Y


def find_start_points(
    fun: Callable[[TensorType], TensorType],
    space: Box,
    num_starts: int = 16,
    num_random_batches: int = 64,
    random_batch_shape: Sequence[int] = (1024,),
    custom_batches: Iterable[TensorType] | None = None,
    num_cmaes_runs: int = 0,
    cmaes_kwargs: dict | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    # Create a list of iterables for starting position candidates
    choices: list[Iterable[tuple[TensorType, TensorType]]] = []
    if num_cmaes_runs:
        cmaes_kwargs = {} if cmaes_kwargs is None else cmaes_kwargs.copy()
        choices.append(
            run_cmaes(fun, space, **cmaes_kwargs)[::-1] for _ in range(num_cmaes_runs)
        )

    if num_random_batches:
        n = tf.reduce_prod(random_batch_shape)
        shape = tf.concat([random_batch_shape, [-1]], axis=0)
        random_batches = (
            tf.reshape(space.sample_sobol(n), shape) for _ in range(num_random_batches)
        )
        choices.append((fun(x), x) for x in random_batches)

    if custom_batches:
        choices.append((fun(x), x) for x in custom_batches)

    if not choices:
        raise ValueError

    # Extract top-k starting points and values
    return reduce_topk(chain.from_iterable(choices), num_starts)[::-1]


def run_gradient_ascent(
    fun: Callable[[TensorType], TensorType],
    space: Box,
    start_points: TensorType,
    scipy_kwargs: dict | None = None
) -> tuple[tf.Tensor, tf.Tensor]:
    with patch("trieste.acquisition.optimizer.spo.minimize", minimize_with_stopping):
        successes, values, points, n_evals = _perform_parallel_continuous_optimization(
            fun,
            space,
            start_points,
            {} if scipy_kwargs is None else scipy_kwargs,
        )

    return points, values


def run_multistart_gradient_ascent(
    fun: Callable[[TensorType], TensorType],
    space: Box,
    num_starts: int = 16,
    num_random_batches: int = 64,
    random_batch_shape: Sequence[int] = (1024,),
    custom_batches: Iterable[TensorType] | None = None,
    num_cmaes_runs: int = 0,
    cmaes_kwargs: dict | None = None,
    scipy_kwargs: dict | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:

    # Find starting points
    start_points, _ = find_start_points(
        fun=fun,
        space=space,
        num_starts=num_starts,
        num_random_batches=num_random_batches,
        random_batch_shape=random_batch_shape,
        custom_batches=custom_batches,
        num_cmaes_runs=num_cmaes_runs,
        cmaes_kwargs=cmaes_kwargs
    )

    # Check if we are optimizing a single function or multiple functions in parallel
    if tf.rank(start_points) == 3:
        single_problem = False
    elif tf.rank(start_points) == 2:
        single_problem = True
    else:
        raise ValueError(
            f"Expected rank of `start_points` to be two or three, "
            f"but {tf.rank(start_points)=}."
        )

    if single_problem:  # add explicit problem dimension
        start_points = tf.expand_dims(start_points, axis=-2)

    # Run gradient ascent from each starting point
    final_points, final_values = run_gradient_ascent(
        fun=fun,
        space=space,
        start_points=start_points,
        scipy_kwargs=scipy_kwargs,
    )

    if single_problem:  # remove problem dimension
        final_points = tf.squeeze(final_points, axis=1)
        final_values = tf.squeeze(final_values, axis=1)

    # Return the best point and value
    indices = tf.argmax(final_values, axis=-1)
    batch_dims = 0 if single_problem else 1
    best_point = tf.gather(final_points, indices, axis=-2, batch_dims=batch_dims)
    best_value = tf.gather(final_values, indices, axis=-1, batch_dims=batch_dims)
    return best_point, best_value
