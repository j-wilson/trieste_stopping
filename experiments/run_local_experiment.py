from __future__ import annotations

import logging
from argparse import Namespace
from functools import partial
from json import loads
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence non-critical TF logging

import tensorflow as tf
import tensorflow_probability as tfp
import trieste
import trieste_stopping
from gpflow.config import set_default_likelihood_positive_minimum
from trieste.acquisition import AcquisitionRule
from trieste.objectives import SingleObjectiveTestProblem
from trieste_stopping.models import build_model
from trieste_stopping.stopping import StoppingRule

from . import objectives, utils
from .experiment import Experiment, StepData

logger = logging.getLogger()


def get_components(args: Namespace) -> tuple[
    SingleObjectiveTestProblem, AcquisitionRule, StoppingRule
]:
    """Creates the experiment's core components."""
    try:
        kwargs = loads(args.objective_kwargs) if args.objective_kwargs else {}
        objective = getattr(objectives, args.objective)(**kwargs)
    except AttributeError:
        objective = getattr(trieste.objectives, args.objective)

    acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
        builder=trieste_stopping.acquisition.InSampleKnowledgeGradient(),
        optimizer=partial(
            trieste_stopping.acquisition.maximize_acquisition,
            num_cmaes_runs=5,
            cmaes_kwargs={"tolx": 1e-3},
        )
    )

    stopping_rule = utils.get_stopping_rule(
        class_name=args.stopping_rule,
        initial_step=args.initial_step,
        step_limit=args.step_limit,
        acquisition_rule=acquisition_rule,
        **(loads(args.stopping_kwargs) if args.stopping_kwargs else {})
    )

    return objective, acquisition_rule, stopping_rule


def run_local_experiment(args: Namespace) -> tuple[Exception, dict[int, StepData]]:
    """Main method for running an experiment locally."""
    tf.random.set_seed(args.seed)
    set_default_likelihood_positive_minimum(1e-16)

    # Construct the objective, acquisition rule, stopping rule, and link function
    objective, acquisition_rule, stopping_rule = get_components(args)
    link_function = (
        None
        if args.inv_link_function is None
        else tfp.bijectors.Invert(getattr(tfp.bijectors, args.inv_link_function)())
    )

    # Generate initial data
    steps = tuple(range(args.initial_step, args.step_limit))
    query_points = objective.search_space.sample_sobol(steps[0])
    observations = tf.concat(
        [objective.objective(x) for x in tf.split(query_points, steps[0], 0)], axis=0
    )
    initial_data = trieste.data.Dataset(query_points, observations)

    # Run the experiment
    experiment = Experiment(
        objective=objective,
        dataset=initial_data,
        acquisition_rule=acquisition_rule,
        stopping_rule=stopping_rule,
        model=build_model(objective.search_space, initial_data, link_function),
    )
    states = utils.local_runner(
        experiment=experiment,
        steps=steps,
        callback=partial(utils.default_logging_callback, logger=logger)
    )
    return experiment, states


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_local_experiment(args=utils.get_argument_parser().parse_args())
