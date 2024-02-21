from __future__ import annotations

import logging
import argparse
from dataclasses import asdict
from functools import partial
from json import loads

import numpy as np
import tensorflow as tf
from gpflow.config import set_default_likelihood_positive_minimum
from trieste.acquisition import AcquisitionRule
from trieste.objectives import SingleObjectiveTestProblem
from trieste_stopping.stopping import StoppingRule
from . import (
    default_factory_mode,
    Experiment,
    Factory,
    FactoryManager,
    local_runner,
    StepData,
    wandb_runner,
)

logger = logging.getLogger()
trieste = FactoryManager("trieste")  # for details, see `tutorials/factories.ipynb`
trieste_stopping = FactoryManager("trieste_stopping")

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--seed",
    help="Seed for TensorFlow's random number generator.",
    type=int,
    required=True,
)
argparser.add_argument(
    "--wandb",
    help="Optional W&B project name; if passed, experiment will be run as a W&B job.",
    type=str,
    required=False,
)
argparser.add_argument(
    "--initial_step",
    help="Initial step; dictates the number of random trials used to initialize.",
    type=int,
    default=5,
)
argparser.add_argument(
    "--step_limit",
    help="Upper bound on the number of steps.",
    type=int,
    required=True,
)
argparser.add_argument(
    "--problem",
    help="Class name for the problem from trieste[_stopping].objectives.",
    type=str,
    required=True
)
argparser.add_argument(
    "--problem_kwargs",
    help="String encoded dictionary of arguments for the problem.",
    type=str,
    required=False,
)
argparser.add_argument(
    "--stopping_rule",
    help="Class name for the stopping rule to use.",
    type=str,
    default="ProbabilisticRegretBound",
    choices=(
        "FixedBudget", "AcquisitionThreshold", "ConfidenceBound", "ProbabilisticRegretBound"
    ),
)
argparser.add_argument(
    "--stopping_kwargs",
    help="String encoded dictionary of arguments for the stopping rule.",
    type=str,
    required=False,
)


def main():
    args = argparser.parse_args()
    tf.random.set_seed(args.seed)
    set_default_likelihood_positive_minimum(1e-16)

    # Create factories for experiment components
    steps = tuple(range(args.initial_step, args.step_limit))
    problem_factory, acquisition_factory, stopping_factory = create_factories(args)
    if args.wandb is None:
        runner = local_runner
    else:
        config = {
            "seed": args.seed,
            "steps": steps,
            "problem": asdict(problem_factory),
            "acquisition_rule": asdict(acquisition_factory),
            "stopping_rule": asdict(stopping_factory),
        }
        runner = partial(wandb_runner, project=args.wandb, config=config)

    # Generate initial data
    problem = problem_factory()
    num_initial = steps[0]
    query_points = problem.search_space.sample_sobol(num_initial)
    observations = tf.concat(
        [problem.objective(x) for x in tf.split(query_points, num_initial, 0)], axis=0
    )
    initial_data = trieste.data.Dataset(query_points, observations)

    # Run the experiment
    def logger_callback(_: Experiment, state: StepData) -> None:
        logger.info(f"Summarizing step {state.step}...")
        logger.info(f"Incumbent mean: {np.squeeze(state.stopping.incumbent.mean):.2e}")
        logger.info(f"Stopping value: {np.nanmin(state.stopping.value):.2e}")
        logger.info(f"Stopping runtime: {state.stopping.setup_time:.2e}\n")

    experiment = Experiment(
        problem=problem,
        dataset=initial_data,
        acquisition_rule=acquisition_factory(),
        stopping_rule=stopping_factory(),
    )
    states = runner(experiment=experiment, steps=steps, callback=logger_callback)
    return experiment, states


def create_factories(args: argparse.Namespace) -> tuple[
    Factory[SingleObjectiveTestProblem],
    Factory[AcquisitionRule],
    Factory[StoppingRule]
]:
    """Constructs factories for each of the experiment's components. Using Factory
    instances allows us to serialize experiment parameters as W&B job configs."""
    with default_factory_mode.ctx(True):
        # Factory for the minimization objective
        try:
            problem = getattr(trieste_stopping.objectives, args.problem)
        except AttributeError:
            problem = getattr(trieste.objectives, args.problem)

        if args.problem_kwargs:  # update default arguments
            problem = problem(**loads(args.problem_kwargs))

        # Factory for the acquisition rule
        acquisition_rule = trieste.acquisition.rule.EfficientGlobalOptimization(
            builder=trieste_stopping.acquisition.InSampleKnowledgeGradient(),
            optimizer=trieste_stopping.acquisition.maximize_acquisition(
                num_cmaes_runs=5, cmaes_kwargs={"tolx": 1e-3},
            ).as_partial()  # do not "build" this object, retain it as a partial method
        )

        # Factory for the stopping rule
        constructor = getattr(trieste_stopping.stopping, args.stopping_rule)
        if args.stopping_rule == "FixedBudget":
            steps = tuple(range(args.initial_step, args.step_limit))
            stopping_rule = constructor(budget=len(steps))
        elif args.stopping_rule == "AcquisitionThreshold":
            stopping_rule = constructor(acquisition_rule=acquisition_rule, threshold=1e-5)
        elif args.stopping_rule == "ConfidenceBound":
            try:  # ugly...
                dim = problem.wrapped.dim
            except AttributeError:
                dim = problem.keywords["dim"]
            module = trieste_stopping.stopping.criteria.confidence_bound
            schedule = module.build_default_beta_schedule(risk_tolerance=0.05, dim=int(dim))
            stopping_rule = constructor(regret_bound=0.1, beta_schedule=schedule)
        elif args.stopping_rule == "ProbabilisticRegretBound":
            stopping_rule = constructor(
                regret_bound=0.1,  # tolerated amount of regret
                risk_tolerance_model=0.025,  # tolerated risk under the model
                risk_tolerance_error=0.025,  # tolerated risk due to estimation error
                popsize_limit=1000,
                use_unconverged_estimates=1,  # allow use of unconverged estimates
            )

        if args.stopping_kwargs:  # update default arguments
            stopping_rule = stopping_rule(**loads(args.stopping_kwargs))

    return problem, acquisition_rule, stopping_rule


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
