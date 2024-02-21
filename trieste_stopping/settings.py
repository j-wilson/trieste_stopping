from __future__ import annotations

from contextlib import contextmanager
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


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


# Settings for default (hyper)priors
empirical_variance_floor: Setting[float] = Setting(1e-6)
mean_percentile_range: Setting[tuple[float, float]] = Setting((5.0, 95.0))
kernel_lengthscale_median: Setting[float] = Setting(0.5)
kernel_variance_init: Setting[float] = Setting(1.0)
kernel_variance_range: Setting[tuple[float, float]] = Setting((0.1, 10.0))
likelihood_variance_range: Setting[tuple[float, float]] = Setting((1e-9, 10.0))
likelihood_variance_init: Setting[float] = Setting(0.1)

# Settings for trajectory samplers
default_num_features: Setting[int] = Setting(default=1024)
