from __future__ import annotations

import tensorflow as tf
from gpflow.config import default_float
from trieste.acquisition import AcquisitionFunction, AcquisitionRule
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.interfaces import ProbabilisticModel, SupportsGetInternalData
from trieste.space import Box, SearchSpace
from trieste.utils import Timer
from trieste_stopping.incumbent import IncumbentRule
from trieste_stopping.stopping.interface import (
    StoppingCriterion,
    StoppingData,
    StoppingRule,
)


class AcquisitionThreshold(StoppingRule["acquisition_threshold"]):
    def __init__(
        self,
        acquisition_rule: AcquisitionRule,
        threshold: float,
        incumbent_rule: IncumbentRule | None = None,
    ):
        super().__init__(incumbent_rule=incumbent_rule)
        self.acquistion_rule = acquisition_rule
        self.threshold = threshold

    def prepare_stopping_criterion(
        self, model: GaussianProcessRegression, dataset: Dataset,
    ) -> acquisition_threshold:
        return acquisition_threshold(
            model=model,
            incumbent_rule=self.incumbent_rule,
            acquisition_rule=self.acquistion_rule,
            threshold=self.threshold
        )

    def update_stopping_criterion(
        self,
        criterion: acquisition_threshold,
        model: GaussianProcessRegression,
        dataset: Dataset,
    ) -> acquisition_threshold:
        if not isinstance(criterion, acquisition_threshold):
            raise TypeError

        if model is not criterion.model:
            raise NotImplementedError

        if self.incumbent_rule not in (None, criterion.incumbent_rule):
            raise NotImplementedError

        criterion.acquisition_rule = self.acquistion_rule
        criterion.threshold.assign(self.threshold)
        return criterion


class acquisition_threshold(StoppingCriterion):
    def __init__(
        self,
        model: SupportsGetInternalData,
        acquisition_rule: AcquisitionRule,
        threshold: float,
        incumbent_rule: IncumbentRule | None = None,
    ):
        super().__init__(model=model, incumbent_rule=incumbent_rule)
        self.acquisition_rule = acquisition_rule
        self.threshold = tf.Variable(threshold, dtype=default_float())

    def __call__(self, space: SearchSpace, dataset: Dataset) -> StoppingData:
        """
        Tests whether there exists a point whose acquisition value is greater
        than or equal to a given threshold.
        """
        with Timer() as timer:
            value = self.evaluate(space=space, dataset=dataset)
            incumbent = self.incumbent_rule(
                model=self.model, space=space, dataset=dataset
            )

        return StoppingData(
            done=value < self.threshold,
            value=value,
            incumbent=incumbent,
            setup_time=timer.time,
        )

    def evaluate(self, space: SearchSpace, dataset: Dataset) -> tf.Tensor:
        """Returns the maximum acquisition value."""
        point = self.acquisition_rule.acquire_single(
            model=self.model,
            dataset=self.model.get_internal_data(),
            search_space=space,
        )
        return tf.squeeze(self.acquisition_function(point[:, None, :]), -1)

    @property
    def acquisition_function(self) -> AcquisitionFunction | None:
        return self.acquisition_rule._acquisition_function
