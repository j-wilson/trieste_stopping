from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import tensorflow as tf
from trieste_stopping.models import build_model, get_parameters, set_parameters
from trieste_stopping.stopping.interface import StoppingData, StoppingRule
from trieste.acquisition.interface import AcquisitionFunction
from trieste.acquisition.rule import AcquisitionRule
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.objectives import SingleObjectiveTestProblem
from trieste.types import TensorType
from trieste.utils import Timer
from trieste_stopping.utils import PointData


@dataclass
class ModelData:
    parameters: dict[str, TensorType]
    setup_time: float | None = None


@dataclass
class QueryData(PointData):
    point: TensorType
    index: int | None
    mean: TensorType | None
    variance: TensorType | None
    observation: TensorType | None = None
    acquisition: TensorType | None =  None
    setup_time: float | None = None
    observation_time: float | None = None


@dataclass
class StepData:
    step: int
    query: QueryData | None = None
    model: ModelData | None = None
    stopping: StoppingData | None = None


class Experiment:
    def __init__(
        self,
        objective: SingleObjectiveTestProblem,
        dataset: Dataset,
        acquisition_rule: AcquisitionRule,
        stopping_rule: StoppingRule,
        model: GaussianProcessRegression | None = None,
    ):
        if model is None:
            model = build_model(objective.search_space, dataset)

        self.objective = objective
        self.optimizer = AskTellOptimizer(
            models=model,
            datasets=dataset,
            search_space=objective.search_space,
            acquisition_rule=acquisition_rule,
            fit_model=False,
        )
        self.stopping_rule = stopping_rule

    def step(
        self,
        step: int,
        dataset: Dataset | None = None,
        fit_model: bool | None = None,
        parameters: dict | None = None,
    ) -> StepData:
        # Update the experiment
        model_data = self.update(dataset, fit_model=fit_model, parameters=parameters)

        # Evaluate the stopping rule
        stopping_data = self.stopping_rule(
            model=self.model, space=self.objective.search_space, dataset=self.dataset,
        )

        # Maybe ask-tell another query
        if stopping_data.done:
            query_data = None
        else:
            query_data = self.ask(step)
            with Timer() as timer:
                query_data.observation = self.objective.objective(query_data.point)
            query_data.observation_time = timer.time
            self.tell(new_data=Dataset(query_data.point, query_data.observation))

        return StepData(
            step=step,
            query=query_data,
            model=model_data,
            stopping=stopping_data,
        )

    def update(
        self,
        dataset: Dataset | None = None,
        fit_model: bool | None = None,
        parameters: dict | None = None,
    ) -> ModelData:
        if dataset is None:
            dataset = self.dataset

        with Timer() as timer:
            self.model.update(dataset)
            if parameters is not None:
                set_parameters(self.model.model, parameters)

            if fit_model or (fit_model is None and parameters is None):
                self.model.optimize(dataset)

        if parameters is None:
            parameters = {
                name: tf.convert_to_tensor(param)
                for name, param in get_parameters(self.model.model)
            }
        return ModelData(parameters=parameters, setup_time=timer.time)

    def ask(self, step: int) -> QueryData:
        with Timer() as timer:
            point = self.optimizer.ask()

        mean, variance = self.model.predict(point)
        acquisition = tf.squeeze(self.acquisition_function(point[None]))
        return QueryData(
            point=point,
            index=step,
            mean=mean,
            variance=variance,
            acquisition=acquisition,
            setup_time=timer.time,
        )

    def tell(self, new_data: Dataset) -> None:
        """Wrapper for `tell` that skips updating the model."""
        with (
            patch.object(self.optimizer.model, "update"),
            patch.object(self.optimizer.model, "optimize"),
        ):
            self.optimizer.tell(new_data)

    @property
    def acquisition_function(self) -> AcquisitionFunction | None:
        return self.optimizer._acquisition_rule._acquisition_function

    @property
    def dataset(self) -> Dataset:
        return self.optimizer.dataset

    @property
    def model(self) -> GaussianProcessRegression:
        return self.optimizer.model
