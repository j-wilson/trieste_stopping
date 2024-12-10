from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import asdict
from functools import partial
from json import loads
from os import environ
from unittest.mock import patch
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence non-critical TF logging

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.config import set_default_likelihood_positive_minimum
from trieste.acquisition import AcquisitionRule
from trieste.objectives import SingleObjectiveTestProblem
from trieste_stopping.models import build_model
from trieste_stopping.stopping import StoppingRule

from . import objectives, utils
from .experiment import Experiment
from .factories import default_factory_mode, Factory, FactoryManager

logger = logging.getLogger()
objectives = FactoryManager(objectives)
trieste = FactoryManager("trieste")  # for details, see `tutorials/factories.ipynb`
trieste_stopping = FactoryManager("trieste_stopping")


def get_factories(args: Namespace) -> tuple[
    Factory[SingleObjectiveTestProblem], Factory[AcquisitionRule], Factory[StoppingRule]
]:
    """Creates factories for the experiment's core components.

    The first context manager tells the FactoryManagers to return factories that wrap
    the underlying attributes. The second context manager temporarily overwrites the
    `trieste_stopping` package used in `utils` with the FactoryManager defined above.
    """
    with (
        default_factory_mode.ctx(True),  # tell FactoryManagers to produce factories
        patch.object(utils, "trieste_stopping", trieste_stopping)  # use the FactoryManager
    ):
        # Factory for the minimization objective
        try:
            kwargs = loads(args.objective_kwargs) if args.objective_kwargs else {}
            objective = getattr(objectives, args.objective)(**kwargs)
        except AttributeError:
            objective = getattr(trieste.objectives, args.objective)

        # Factory for the acquisition rule
        acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
            builder=trieste_stopping.acquisition.InSampleKnowledgeGradient(),
            optimizer=trieste_stopping.acquisition.maximize_acquisition(
                num_cmaes_runs=5, cmaes_kwargs={"tolx": 1e-3},
            ).as_partial()  # this is a factory for a partial method, not a class!
        )

        stopping_rule = utils.get_stopping_rule(
            class_name=args.stopping_rule,
            initial_step=args.initial_step,
            step_limit=args.step_limit,
            acquisition_rule=acquisition_rule,
            **(loads(args.stopping_kwargs) if args.stopping_kwargs else {})
        )

    return objective, acquisition_rule, stopping_rule


def run_wandb_experiment(args: Namespace):
    tf.random.set_seed(args.seed)
    set_default_likelihood_positive_minimum(1e-16)

    # Create factories for experiment components and serialize as W&B run config
    steps = tuple(range(args.initial_step, args.step_limit))
    objective_factory, acquisition_factory, stopping_factory = get_factories(args)
    wandb_config = {
        "seed": args.seed,
        "steps": steps,
        "objective": asdict(objective_factory),
        "acquisition_rule": asdict(acquisition_factory),
        "stopping_rule": asdict(stopping_factory),
        "inv_link_function": args.inv_link_function,
    }

    # Invoke the factories to get the actual instances
    objective = objective_factory()
    acquisition_rule = acquisition_factory()
    stopping_rule = stopping_factory()
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

    experiment = Experiment(
        objective=objective,
        dataset=initial_data,
        acquisition_rule=acquisition_rule,
        stopping_rule=stopping_rule,
        model=build_model(objective.search_space, initial_data, link_function),
    )

    states = utils.wandb_runner(
        project=args.project,
        config=wandb_config,
        experiment=experiment,
        steps=steps,
        callback=partial(utils.default_logging_callback, logger=logger),
    )
    return experiment, states


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = utils.get_argument_parser()
    parser.add_argument(
        "--project",
        help="Name of an existing W&B project.",
        type=str,
        required=True,
    )
    run_wandb_experiment(args=parser.parse_args())
