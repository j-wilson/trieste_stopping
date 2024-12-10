from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, Iterator, Sequence, TypeVar

from trieste.types import TensorType


T = TypeVar("T")


@dataclass
class PointData:
    point: TensorType
    index: int | Sequence[int] | None
    mean: TensorType | None
    variance: TensorType | None
    observation: TensorType | None = None


class Setting(Generic[T]):
    def __init__(self, default: T) -> None:
        self.default = default

    def get(self) -> T:
        return self.default

    def set(self, value: T) -> None:
        self.default = value

    @contextmanager
    def ctx(self, value: T) -> Iterator[None]:
        prev = self.get()
        try:
            self.set(value)
            yield
        finally:
            self.set(prev)

    __call__ = get
