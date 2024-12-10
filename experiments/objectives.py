from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt
from numbers import Number
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from gpflow.base import TensorType
from gpflow.config import default_float
from gpflow.kernels import Matern52
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from pandas import get_dummies
from trieste.space import Box
from trieste_stopping.models.feature_maps import draw_kernel_feature_map
from trieste_stopping.models.trajectories import LinearTrajectory
from trieste_stopping.utils.spaces import NamedSpaceWrapper, UnitHypercube
from xgboost import XGBClassifier

ModelType = TypeVar("ModelType")
SupervisedDataset = tuple[TensorType, TensorType]


class Matern52Trajectory:
    """An (approximate) draw from a Gaussian process prior with a Matern-5/2 kernel."""

    def __init__(
        self,
        dim: int,
        lengthscales: Number | Sequence[Number],
        noise_variance: float = 0.0,
        num_features: int = 4096,
    ):
        if isinstance(lengthscales, Number):
            lengthscales = [lengthscales for _ in range(dim)]

        feature_map = draw_kernel_feature_map(
            kernel=Matern52(lengthscales=lengthscales),
            num_inputs=dim,
            num_features=num_features,
        )
        self.noise_variance = noise_variance
        self.search_space = Box(lower=dim * [0], upper=dim * [1])
        self.trajectory = LinearTrajectory(
            feature_map=feature_map,
            weights=tf.random.normal([num_features, 1], dtype=default_float())
        )

    def objective(self, x: tf.Tensor, noisy: bool = True) -> tf.Tensor:
        values = self.trajectory(x)
        if noisy:
            rvs = tf.random.normal(shape=values.shape, dtype=values.dtype)
            values += sqrt(self.noise_variance) * rvs

        return values

    @property
    def dim(self) -> int:
        """The input dimensionality of the test function"""
        return self.search_space.dimension

    @property
    def bounds(self) -> list[list[float]]:
        """The input space bounds of the test function"""
        return [self.search_space.lower, self.search_space.upper]


class AutoMLObjective(ABC, Generic[ModelType]):
    """Base class for AutoML objectives."""
    search_space: NamedSpaceWrapper[UnitHypercube]

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._test_set = None
        self._train_set = None

    def objective(self, x: TensorType) -> tf.Tensor:
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)

        hyperparameters = self.search_space.to_dict(self.search_space.from_unit(x))
        model = self.train(self.train_set, **hyperparameters)
        value = tf.convert_to_tensor(self.test(model), dtype_hint=default_float())
        return tf.reshape(value, x.shape[:-1] + [1])

    @abstractmethod
    def train(self, **kwargs: Any) -> ModelType:
        pass

    @abstractmethod
    def test(self, model: ModelType) -> float:
        pass

    @abstractmethod
    def get_datasets(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        pass

    def set_datasets(self) -> None:
        self._train_set, self._test_set = self.get_datasets()

    @property
    def train_set(self) -> SupervisedDataset:
        if self._train_set is None:
            self.set_datasets()

        return self._train_set

    @property
    def test_set(self) -> SupervisedDataset:
        if self._test_set is None:
            self.set_datasets()

        return self._test_set

    @classmethod
    @property
    def dim(cls) -> int:
        return cls.search_space.dimension


class MNIST(AutoMLObjective[tf.keras.Model]):
    """A simple CNN classifier on the MNIST dataset."""

    name = "MNIST"
    search_space = NamedSpaceWrapper(
        names=("filters", "epochs", "log_learning_rate", "dropout_rate"),
        space=UnitHypercube([1, 1, np.log(1e-5), 0], [64, 25, np.log(0.1), 1 - 1e-6])
    )

    def __init__(self, seed: int | None = None, verbose: bool = True):
        super().__init__(seed=seed)
        self.verbose = verbose
        self._test_set = None
        self._train_set = None

    def test(self, model: tf.keras.Model) -> float:
        _, acc = model.evaluate(*self.test_set, verbose=self.verbose)
        return 1 - acc

    def train(
        self,
        dataset: tuple[tf.Tensor, tf.Tensor],
        filters: float | int,
        epochs: float | int,
        log_learning_rate: float,
        dropout_rate: float,
    ) -> tf.keras.Model:
        # Post-process hyperparameters
        _filters = int(np.round(filters))
        _epochs = int(np.round(epochs))
        _learning_rate = float(np.exp(log_learning_rate))
        _dropout_rate = float(dropout_rate)

        # Build and train the model
        input_shape = dataset[0].shape[1:]
        num_outputs = np.squeeze(dataset[1].shape[1:])
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Conv2D(_filters, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(_filters, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(_dropout_rate),
                tf.keras.layers.Dense(num_outputs, activation="softmax"),
            ]
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=_learning_rate),
            metrics=["accuracy"]
        )
        model.fit(*dataset, epochs=_epochs, batch_size=64, verbose=self.verbose)
        return model

    def get_datasets(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        def preprocess(x: TensorType, y: TensorType) -> SupervisedDataset:
            x = tf.reshape(tf.cast(x, tf.float32) / 255, (len(x), 28, 28, 1))
            y = tf.keras.utils.to_categorical(y, num_classes=10)
            return x, y

        train_set, test_set = tf.keras.datasets.mnist.load_data()
        return preprocess(*train_set), preprocess(*test_set)


class Adult(AutoMLObjective[XGBClassifier]):
    """An XGBoost Classifier on the Adult dataset."""
    name = "Adult"
    search_space = NamedSpaceWrapper(
        names=("max_depth", "log_n_estimators", "log_learning_rate"),
        space=UnitHypercube([1, 0, np.log(1e-3)], [10, np.log(1000), 0])
    )

    def test(self, model: XGBClassifier) -> float:
        x_test, y_test = self.test_set
        y_pred = model.predict(x_test)
        return np.mean(y_pred != y_test)

    def train(
        self,
        dataset: SupervisedDataset,
        max_depth: float | int,
        log_n_estimators: float,
        log_learning_rate: float,
    ) -> XGBClassifier:
        # Post-process hyperparameters
        _max_depth = int(np.round(max_depth))
        _n_estimators = int(np.round(np.exp(log_n_estimators)))
        _learning_rate = float(np.exp(log_learning_rate))

        # Build and train the model
        model = XGBClassifier(
            max_depth=_max_depth,
            n_estimators=_n_estimators,
            learning_rate=_learning_rate,
            random_state=self.seed,
        )
        model.fit(*dataset)
        return model

    def get_datasets(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        df = load_dataset("scikit-learn/adult-census-income")["train"].to_pandas()

        y = df.pop("income").astype("category").to_numpy().reshape(-1, 1)
        y = np.squeeze(OrdinalEncoder().fit(y).transform(y), axis=-1)

        df.drop(["education"], axis=1, inplace=True)  # duplicate
        x = get_dummies(df).to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=self.seed
        )
        return (x_train, y_train), (x_test, y_test)
