from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import asdict
from itertools import repeat
from logging import Logger
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import tensorflow as tf
import trieste_stopping
import wandb
from gpflow.base import Parameter
from trieste.acquisition import AcquisitionRule
from trieste_stopping.models import get_link_function
from trieste_stopping.utils import get_expected_value
from trieste_stopping.utils.level_tests import LevelTestConvergence
from trieste_stopping.stopping.interface import StoppingRule

from .experiment import Experiment, StepData


def local_runner(
    experiment: Experiment,
    steps: Iterable[int],
    step_kwargs: Iterable[dict] | None = None,
    callback: Callable[[Experiment, StepData], None] | None = None,
) -> dict[int, StepData]:
    if step_kwargs is None:
        step_kwargs = repeat({})

    states = {}
    for step, kwargs in zip(steps, step_kwargs):
        state = states[step] = experiment.step(step=step, **kwargs)
        if callback:
            callback(experiment, state)

        if state.stopping is not None and state.stopping.done:
            break

    return states


def wandb_runner(
    project: str,
    experiment: Experiment,
    steps: Iterable[int],
    step_kwargs: Iterable[dict] | None = None,
    tags: Sequence[str] | None = None,
    config: dict | None = None,
    callback: Callable[[Experiment, StepData], None] | None = None,
) -> dict[int, StepData]:
    if step_kwargs is None:
        step_kwargs = repeat({})

    states = {}
    with wandb.init(project=project, config=config, tags=tags) as job:
        # Make the artifact (i.e. directory) uploaded at the end
        artifact = wandb.Artifact(name=job.name, type="experiment")

        # Step through the experiment
        try:
            for step, kwargs in zip(steps, step_kwargs):
                state = states[step] = experiment.step(step=step, **kwargs)
                if callback:
                    callback(experiment, state)

                # Upload logs to W&B
                job.log(_state_to_wandb_log(state), step=step)

                # Add the step result to the artifact
                with artifact.new_file(name=f"state_{step}", mode="w") as fp:
                    json.dump(obj=asdict(state), fp=fp, cls=DefaultEncoder)

                if state.stopping is not None and state.stopping.done:
                    break
        finally:
            # Add final dataset to the artifact
            with artifact.new_file(name=f"dataset", mode="w") as fp:
                json.dump(obj=asdict(experiment.dataset), fp=fp, cls=DefaultEncoder)

            # Upload the artifact
            job.log_artifact(artifact)

    return states


class DefaultEncoder(json.JSONEncoder):
    def default(self, obj):
        if tf.is_tensor(obj) or isinstance(obj, Parameter):
            return obj.numpy().tolist()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        return super().default(obj)


def get_stopping_rule(
    class_name: str,
    initial_step: int,
    step_limit: int,
    acquisition_rule: AcquisitionRule | None = None,
    risk_bound: float = 0.05,
    **kwargs,
) -> StoppingRule:
    """
    Helper for creating a StoppgingRule with basic defaults that mirror experiments
    from the paper.

    Args:
        class_name: The name of the class.
        initial_step: The number of queries used to initialize the search
        step_limit: Limit on the number of queries used for search, i.e. a budget.
        risk_bound: Bound on chance that stop conditions do not hold under the model.
        acquisition_rule: An optional acquisition rule used to define the stopping rule.

    Returns: A StoppingRule instance.
    """
    if class_name == "FixedBudget":
        defaults = {"budget": step_limit}
    elif class_name == "AcquisitionThreshold":
        defaults = {"acquisition_rule": acquisition_rule, "threshold": 1e-5}
    elif class_name in ("ConfidenceBound", "ChangeInExpectedMinimum"):
        defaults = {"threshold": 0.1, "risk_bound": risk_bound}
    elif class_name == "ProbabilisticRegretBound":
        defaults = {
            "prob_bound": 0.5 * risk_bound,
            "regret_bound": 0.1,
            "risk_schedule": trieste_stopping.utils.schedules.ConstantSchedule(
                constant=0.5 * risk_bound/(step_limit - initial_step)
            ),
            "level_test": trieste_stopping.utils.level_tests.ClopperPearsonLevelTest(
                size_limit=1000,  # maximum number of samples used by test
                convergence=LevelTestConvergence.ANY_LE,
            ),
            "enforce_convergence": 0,
        }

    return getattr(trieste_stopping.stopping, class_name)(**(defaults | kwargs))


def get_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        help="Seed for TensorFlow's random number generator.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--initial_step",
        help="Initial step; dictates the number of random trials used to initialize.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--step_limit",
        help="Upper bound on the number of steps.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--objective",
        help="Class name for the objective from `.objectives` or `trieste.objectives`.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--objective_kwargs",
        help="String encoded dictionary of arguments for the objective.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--stopping_rule",
        help="Class name for the stopping rule to use.",
        type=str,
        default="FixedBudget",
    )
    parser.add_argument(
        "--stopping_kwargs",
        help="String encoded dictionary of arguments for the stopping rule.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--inv_link_function",
        help="Class name of a Bijector to use as an inverse link function.",
        type=str,
        required=False,
    )

    return parser


def default_logging_callback(experiment: Experiment, state: StepData, logger: Logger):
    bridge = experiment.model
    model = bridge.model
    link = get_link_function(bridge, skip_identity=True)

    # Summarize the model
    logger.info(f"Summarizing step {state.step}...")
    logger.info(f"Prior mean: {np.squeeze(model.mean_function.c):.2e}")
    logger.info(f"Prior variance: {np.squeeze(model.kernel.variance):.2e}")
    logger.info(f"Prior lengthscales: {np.squeeze(model.kernel.lengthscales)}")
    logger.info(f"Likelihood variance: {np.squeeze(model.likelihood.variance):.2e}")

    # Summarize the stoppping rule and best point
    best = state.stopping.best_point
    logger.info(f"Stopping value: {state.stopping.value:.2e}")
    logger.info(f"Stopping runtime: {state.stopping.setup_time:.2e}")
    logger.info(f"Best point x{best.index}: {np.squeeze(best.point)}")
    if link is None:
        logger.info(f"Best point mean: {np.squeeze(best.mean):.2e}")
    else:
        native = get_expected_value(best.mean, best.variance, inverse=link)
        logger.info(f"Best point mean (linked): {np.squeeze(best.mean):.2e}")
        logger.info(f"Best point mean (native): {np.squeeze(native):.2e}")

    # Summarize the query
    query = state.query
    if query:
        logger.info(f"Query point x{query.index}: {np.squeeze(query.point)}")
        logger.info(f"Query acquisition: {np.squeeze(query.acquisition):.2e}")
        if link is None:
            logger.info(f"Query mean: {np.squeeze(query.mean):.2e}")
        else:
            native = get_expected_value(query.mean, query.variance, inverse=link)
            logger.info(f"Query mean (linked): {np.squeeze(query.mean):.2e}")
            logger.info(f"Query mean (native): {np.squeeze(native):.2e}")
        logger.info(f"Query observation: {np.squeeze(query.observation):.2e}")


def flatten_dict(
    src: dict,
    include: Callable[[str, Any], bool] | None = None,
    exclude: Callable[[str, Any], bool] | None = None,
    transform: Callable[[str, Any], Iterator[tuple[str, Any]]] | None = None,
    delimiter: str = ".",
) -> Iterator[tuple[str, object]]:
    for key, val in src.items():
        if include and not include(key, val):
            continue

        if exclude and exclude(key, val):
            continue

        if isinstance(val, dict):
            for k, v in flatten_dict(
                val,
                include=include,
                exclude=exclude,
                transform=transform,
                delimiter=delimiter,
            ):
                yield ".".join((key, k)), v
        elif transform:
            yield from transform(key, val)
        else:
            yield key, val


def _state_to_wandb_log(state: StepData) -> dict[str, Any]:
    logs = {}
    for key, val in flatten_dict(
        src=asdict(state),
        exclude=lambda key, val: key in ("step", "point") or val is None,
        transform=_wandb_summary_transform,
        delimiter="."
    ):
        # Group top-level fields together on W&B
        logs[key.replace(".", "/", 1)] = val

    return logs


def _wandb_summary_transform(
    key: str, val: Any, percentiles: Sequence[float] = (0.0, 0.5, 1.0),
) -> Iterator[tuple[str, Any]]:
    """Converts items with multiple (or variadic) values into multiple items,
    corresponding to percentiles of the origin item."""
    if not isinstance(val, (np.ndarray, tf.Tensor, tf.Variable, Parameter)):
        yield key, val
        return

    if val.shape is None or None in val.shape or np.prod(val.shape) > 1:
        for p, q in zip(percentiles, np.nanquantile(val, q=percentiles)):
            yield f"{key}_p{p}", q
    else:
        yield key, np.squeeze(val)
