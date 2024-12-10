from __future__ import annotations

import inspect
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass, InitVar
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Callable, Generic, Iterator, overload, Sequence, TypeVar
from typing_extensions import Self

from multipledispatch import Dispatcher
from trieste_stopping.utils import Setting

T = TypeVar("T", bound=object)
Build = Dispatcher("build")
default_factory_mode: Setting[bool] = Setting(default=False)


def get_module(path: str) -> ModuleType:
    if path in sys.modules:
        return sys.modules[path]

    spec = find_spec(path)
    return spec.loader.load_module()


class FactoryModeMixin:
    factory: bool | None

    def staging(self) -> bool:
        return default_factory_mode() if self.factory is None else self.factory

    @contextmanager
    def factory_ctx(self, mode: bool) -> Iterator[None]:
        prev = self.factory
        try:
            self.factory = mode
            yield
        finally:
            self.factory = prev


class Factory(Generic[T], FactoryModeMixin, ABC):
    path: str  # qualified name of the target object
    factory: bool | None
    wrapped: Callable[..., T] | T | None  # this attribute should not be serialized

    @abstractmethod
    def __call__(self, *args, **kwargs) -> T | Factory[T]:
        pass


@dataclass
class Loader(Factory[T]):
    path: str  # qualified name of the target object
    factory: bool | None = None
    wrapped: InitVar[T | None] = None  # does not get serialized

    def __post_init__(self, wrapped: T | None) -> None:
        if "." not in self.path:
            raise ValueError

        if wrapped is None:
            *src, name = self.path.split(".")
            module = get_module(".".join(src))
            wrapped = getattr(module, name)

        self.wrapped: T = wrapped

    def __call__(self, *args: Any, **kwargs: Any) -> T | Builder[T]:
        if self.staging():
            if not isinstance(self.wrapped, Callable):
                raise RuntimeError(
                    f"Attempted to call a {type(self)} for a {type(self.wrapped)}-typed"
                    f" object (non-callable) while in factory mode."
                )

            return Builder(
                path=self.path,
                factory=self.factory,
                wrapped=self.wrapped,
                args=list(args),
                keywords=kwargs,
            )

        if args or kwargs:
            raise TypeError(
                f"{type(self)} does not accept arguments when called outside of factory"
                f" mode."
            )

        return self.wrapped


@dataclass
class Partial(Factory[T]):
    path: str  # qualified name of the target object
    factory: bool | None = None
    wrapped: InitVar[Callable[..., T] | None] = None  # does not get serialized
    args: list[Any] = field(default_factory=list)
    keywords: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self, wrapped: Callable[..., T] | None) -> None:
        if "." not in self.path:
            raise ValueError

        if wrapped is None:
            *src, name = self.path.split(".")
            module = get_module(".".join(src))
            wrapped = getattr(module, name)

        if not isinstance(wrapped, Callable):
            raise TypeError(f"Wrapped type {type(wrapped)} is not callable.")

        keywords = {}  # get defaults for all optional parameters
        var_keywords = False
        for name, param in inspect.signature(wrapped).parameters.items():
            var_keywords = var_keywords or param.kind is param.VAR_KEYWORD
            if name in self.keywords or param.default is not inspect._empty:
                keywords[name] = self.keywords.pop(name, param.default)

        if self.keywords and not var_keywords:
            raise ValueError(
                f"Found unexpected keyword arguments for {wrapped}: "
                f"{', '.join(list(self.keywords))}."
            )

        self.keywords = keywords
        self.wrapped: Callable[..., T] = wrapped

    def __call__(self, *args: Any, **kwargs: Any) -> T | type[Self]:
        if self.staging():
            return type(self)(
                path=self.path,
                factory=self.factory,
                wrapped=self.wrapped,
                args=self.args + list(args),
                keywords=self.keywords | kwargs
            )

        args = build(tuple(self.args) + args)
        kwargs = build(self.keywords | kwargs)
        return self.wrapped(*args, **kwargs)


class Builder(Partial[T]):
    def as_partial(self) -> Partial[T]:
        return Partial(
            path=self.path,
            factory=self.factory,
            args=self.args,
            keywords=self.keywords,
            wrapped=self.wrapped
        )


@dataclass
class FactoryManager(FactoryModeMixin):
    """
    Wrapper for Python modules that facilitates the construction of factories.

    When not in factory mode, attributes access return of the attributes of the wrapped
    module. When in factory mode, attribute access returns a FactoryManager if the
    attribute is a Module and a Loader otherwise.
    """
    wrapped: InitVar[ModuleType | str]
    factory: bool | None = None

    def __post_init__(self, wrapped: ModuleType | str):
        if isinstance(wrapped, FactoryManager):
            raise TypeError(f"Cannot recursively wrap {type(self)}.")

        if isinstance(wrapped, str):
            wrapped: ModuleType = get_module(wrapped)

        self.wrapped: ModuleType = wrapped

    def __getattr__(self, name: str) -> FactoryManager | Loader:
        obj = getattr(self.wrapped, name)
        if not self.staging():
            return obj

        if inspect.ismodule(obj):
            return FactoryManager(wrapped=obj, factory=self.factory)

        path = ".".join((self.wrapped.__name__, name))
        return Loader(path=path, factory=self.factory, wrapped=obj)


@overload
def build(obj: Builder[T]) -> T:
    pass


@overload
def build(obj: Loader[T]) -> T:
    pass


@overload
def build(obj: T) -> T:
    pass


def build(obj):
    return Build(obj)


@Build.register((Builder, Loader))
def _(obj: Builder[T] | Loader[T]) -> T:
    with obj.factory_ctx(False):
        return obj()


@Build.register((list, set, tuple))
def _(obj: Sequence) -> Sequence:
    iterator = (build(val) for val in obj)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*iterator)  # check for NamedTuples

    return type(obj)(iterator)


@Build.register(object)
def _(obj: T) -> T:
    if is_dataclass(obj):  # check for dataclasses
        return type(obj)(**build(obj.__dict__))

    return obj


@Build.register(dict)
def _(obj: dict,) -> dict:
    return {key: build(val) for key, val in obj.items()}
