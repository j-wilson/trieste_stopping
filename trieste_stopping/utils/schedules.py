from abc import abstractmethod
from math import log, pi
from typing import Generic, TypeVar

import tensorflow as tf
from gpflow.base import default_float

T = TypeVar("T")


class Schedule(Generic[T]):
    @abstractmethod
    def __call__(self, step: int) -> T:
        pass


class ConstantSchedule(Schedule):
    """Returns f(t) = c."""
    def __init__(self, constant: T) -> None:
        self.constant = constant

    def __call__(self, step: int) -> T:
        return self.constant


class GeometricSchedule(Schedule):
    """Returns f(t) = a * b^{t - 1}, where a, b \in R and t \in N."""
    def __init__(self, base: T, initial_value: T = 1.0):
        self.base = base
        self.initial_value = initial_value

    def __call__(self, step: int) -> T:
        return self.initial_value * self.base ** (step - 1)


class PowerSchedule(Schedule):
    """Returns f(t) = a * t^b, where a, b \in R and t \in N."""
    def __init__(self, power: T, initial_value: T = 1.0):
        self.power = power
        self.initial_value = initial_value

    def __call__(self, step: int) -> T:
        return self.initial_value * (step ** self.power)


class BetaParameterSchedule(Schedule[float]):
    """Default schedule for UCB beta parameter."""
    def __init__(self, cardinality: int, risk_bound: float, scale: float = 1.0):
        self.cardinality = cardinality
        self.risk_bound = risk_bound
        self.scale = scale

    def __call__(self, step: int) -> float:
        return self.scale * (
            log((step * pi) ** 2 * self.cardinality / (6 * self.risk_bound))
        )
