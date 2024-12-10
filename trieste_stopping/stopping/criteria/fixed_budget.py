from __future__ import annotations

from typing import Any

import tensorflow as tf
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box, SearchSpace
from trieste.utils import Timer
from trieste_stopping.selection import SelectionRule
from trieste_stopping.stopping.interface import (
    StoppingCriterion,
    StoppingData,
    StoppingRule,
)


class FixedBudget(StoppingRule["fixed_budget"]):
    """
    Stopping rule that terminates when a predefined number of queires have been made.
    """

    def __init__(self, budget: int, best_point_rule: SelectionRule | None = None):
        """
        Args:
            budget: Upper bound on the number of queries allowed.
            best_point_rule: A rule for choosing a most-preferred point.
        """
        super().__init__(best_point_rule=best_point_rule)
        self._budget = budget

    def prepare_stopping_criterion(
        self, model: Any, dataset: Dataset,
    ) -> fixed_budget:
        return fixed_budget(
            model=model, best_point_rule=self._best_point_rule, budget=self._budget
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

        if self._best_point_rule not in (None, criterion.best_point_rule):
            raise NotImplementedError

        criterion.budget = self._budget
        return criterion


class fixed_budget(StoppingCriterion):
    def __init__(
        self,
        model: GaussianProcessRegression,
        budget: int,
        best_point_rule: SelectionRule | None,
    ):
        super().__init__(model=model, best_point_rule=best_point_rule)
        self.budget = budget

    def __call__(self, space: SearchSpace, dataset: Dataset) -> StoppingData:
        with Timer() as timer:
            done = bool(tf.shape(dataset.observations)[0] >= self.budget)
            best_point = self.best_point_rule(
                model=self.model, space=space, dataset=dataset
            )

        return StoppingData(
            done=done, value=float("nan"), best_point=best_point, setup_time=timer.time
        )
