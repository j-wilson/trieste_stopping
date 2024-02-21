from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import Any, Generic, TypeVar

from trieste.data import Dataset
from trieste.models.interfaces import ProbabilisticModelType
from trieste.space import SearchSpace
from trieste.types import TensorType
from trieste_stopping.incumbent import (
    IncumbentData,
    IncumbentRule,
    InSamplePosteriorMinimizer
)

StoppingCriterionType = TypeVar("StoppingCriterionType", bound="StoppingCriterion")


@dataclass
class StoppingData:
    done: bool | int
    value: Number | TensorType
    incumbent: IncumbentData
    setup_time: float

    def __post_init__(self):
        self.done = int(self.done)
        if self.done < 0 or self.done > 1:
            raise ValueError


class StoppingCriterion(Generic[ProbabilisticModelType], ABC):
    """Interface for a generic stopping criterion."""
    def __init__(
        self,
        model: ProbabilisticModelType,
        incumbent_rule: IncumbentRule | None = None,
    ):
        self.model = model
        self.incumbent_rule = incumbent_rule or InSamplePosteriorMinimizer()

    @abstractmethod
    def __call__(self, space: SearchSpace, dataset: Dataset) -> StoppingData:
        pass

    @abstractmethod
    def evaluate(self, space: SearchSpace, dataset: Dataset) -> Any:
        pass


class StoppingRule(Generic[StoppingCriterionType]):
    """Interface for a generic stopping rule."""

    def __init__(self, incumbent_rule: IncumbentRule | None = None):
        self.incumbent_rule = incumbent_rule or InSamplePosteriorMinimizer()
        self._criterion = None

    def __call__(
        self,
        model: Any,
        space: SearchSpace,
        dataset: Dataset,
        **kwargs: Any,
    ) -> StoppingData:
        if self.criterion is None:
            self.criterion = self.prepare_stopping_criterion(
                model=model,
                dataset=dataset,
            )
        else:
            self.criterion = self.update_stopping_criterion(
                criterion=self.criterion, model=model, dataset=dataset
            )

        return self.criterion(space=space, dataset=dataset, **kwargs)

    @abstractmethod
    def prepare_stopping_criterion(
        self,
        model: Any,
        dataset: Dataset,
    ) -> StoppingCriterionType:
        pass

    @abstractmethod
    def update_stopping_criterion(
        self,
        criterion: StoppingCriterion,
        model: Any,
        dataset: Dataset,
    ) -> StoppingCriterionType:
        pass

    @property
    def criterion(self) -> StoppingCriterionType | None:
        return self._criterion

    @criterion.setter
    def criterion(self, new: StoppingCriterionType) -> None:
        self._criterion = new
