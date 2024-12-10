from __future__ import annotations

import tensorflow as tf
from gpflow.config import default_float
from trieste.acquisition import AcquisitionFunction, AcquisitionRule
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.interfaces import ProbabilisticModel, SupportsGetInternalData
from trieste.space import Box, SearchSpace
from trieste.utils import Timer
from trieste_stopping.selection import SelectionRule
from trieste_stopping.stopping.interface import (
    StoppingData,
    StoppingRule,
    StoppingCriterion,
)


class AcquisitionThreshold(StoppingRule["acquisition_threshold"]):
    """
    Stopping rule that terminates when the maximum acquisition value drops below
    a threshold.
    """

    def __init__(
        self,
        acquisition_rule: AcquisitionRule,
        threshold: float,
        best_point_rule: SelectionRule | None = None,
    ):
        """
        Args:
            acquisition_rule: The acquistion rule used for stopping.
            threshold: A scalar use to decide when to stop.
            best_point_rule: A rule for choosing a most-preferred point.
        """
        super().__init__(best_point_rule=best_point_rule)
        self._acquistion_rule = acquisition_rule
        self._threshold = threshold

    def prepare_stopping_criterion(
        self, model: GaussianProcessRegression, dataset: Dataset,
    ) -> acquisition_threshold:
        return acquisition_threshold(
            model=model,
            best_point_rule=self._best_point_rule,
            acquisition_rule=self._acquistion_rule,
            threshold=self._threshold
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

        if self._best_point_rule not in (None, criterion.best_point_rule):
            raise NotImplementedError

        criterion.acquisition_rule = self._acquistion_rule
        criterion.threshold.assign(self._threshold)
        return criterion


class acquisition_threshold(StoppingCriterion):
    def __init__(
        self,
        model: SupportsGetInternalData,
        acquisition_rule: AcquisitionRule,
        threshold: float,
        best_point_rule: SelectionRule | None = None,
    ):
        super().__init__(model=model, best_point_rule=best_point_rule)
        self.acquisition_rule = acquisition_rule
        self.threshold = tf.Variable(threshold, dtype=default_float())

    def __call__(self, space: SearchSpace, dataset: Dataset) -> StoppingData:
        """
        Tests whether the maximum acquisition value is less-equal to the threshold.
        """
        with Timer() as timer:
            maximizer = self.acquisition_rule.acquire_single(
                model=self.model,
                dataset=self.model.get_internal_data(),
                search_space=space,
            )
            acq_value = tf.squeeze(self.acquisition_function(maximizer[:, None, :]))
            best_point = self.best_point_rule(
                model=self.model, space=space, dataset=dataset
            )

        return StoppingData(
            done=acq_value < self.threshold,
            value=acq_value,
            best_point=best_point,
            setup_time=timer.time,
        )

    @property
    def acquisition_function(self) -> AcquisitionFunction | None:
        return self.acquisition_rule._acquisition_function
