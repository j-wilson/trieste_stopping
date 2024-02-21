from __future__ import annotations

from typing import Any

import tensorflow as tf
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box, SearchSpace
from trieste.utils import Timer
from trieste_stopping.incumbent import IncumbentRule
from trieste_stopping.stopping.interface import (
    StoppingCriterion,
    StoppingData,
    StoppingRule,
)


class FixedBudget(StoppingRule["fixed_budget"]):
    def __init__(self, budget: int,  incumbent_rule: IncumbentRule | None = None):
        super().__init__(incumbent_rule=incumbent_rule)
        self.budget = budget

    def prepare_stopping_criterion(
        self, model: Any, dataset: Dataset,
    ) -> fixed_budget:
        return fixed_budget(
            model=model, incumbent_rule=self.incumbent_rule, budget=self.budget
        )

    def update_stopping_criterion(
        self,
        criterion: fixed_budget,
        model: Any,
        dataset: Dataset,
    ) -> fixed_budget:
        if not isinstance(criterion, fixed_budget):
            raise TypeError

        if model is not criterion.model:
            raise NotImplementedError

        if self.incumbent_rule not in (None, criterion.incumbent_rule):
            raise NotImplementedError

        criterion.budget = self.budget
        return criterion


class fixed_budget(StoppingCriterion):
    def __init__(
        self,
        model: GaussianProcessRegression,
        budget: int,
        incumbent_rule: IncumbentRule | None,
    ):
        super().__init__(model=model, incumbent_rule=incumbent_rule)
        self.budget = budget

    def __call__(self, space: SearchSpace, dataset: Dataset) -> StoppingData:
        with Timer() as timer:
            done = self.evaluate(space=space, dataset=dataset)
            incumbent = self.incumbent_rule(
                model=self.model, space=space, dataset=dataset
            )

        return StoppingData(
            done=done, value=float("nan"), incumbent=incumbent, setup_time=timer.time
        )

    def evaluate(self, space: SearchSpace, dataset: Dataset) -> bool:
        return bool(tf.shape(dataset.observations)[0] >= self.budget)



