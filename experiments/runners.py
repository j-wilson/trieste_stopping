from __future__ import annotations

import json
from dataclasses import asdict
from itertools import repeat
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import tensorflow as tf
import wandb
from gpflow.base import Parameter
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
        exclude=lambda key, val: key == "step" or "point" in key or val is None,
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

    if val.shape == None or None in val.shape or np.prod(val.shape) > 1:
        for p, q in zip(percentiles, np.nanquantile(val, q=percentiles)):
            yield f"{key}_p{p}", q
    else:
        yield key, np.squeeze(val)
