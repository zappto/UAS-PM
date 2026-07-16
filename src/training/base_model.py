"""Abstract base model for all classifiers.

Defines the common interface that all model wrappers must
implement. Provides shared functionality for training time
tracking, metadata collection, and logging.
"""

import time
from abc import ABC, abstractmethod

import numpy as np

from src.config.model_config import ModelConfig


class BaseModel(ABC):
    """Abstract base class for classifier wrappers.

    All model implementations (NB, LR, SVM) must inherit from
    this class and implement the abstract methods.

    Attributes:
        _config: Model training configuration.
        _model: The underlying scikit-learn estimator.
        _training_time: Time taken to train in seconds.
        _is_trained: Whether the model has been trained.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the base model.

        Args:
            config: Training configuration. Uses defaults if None.
        """
        self._config = config or ModelConfig()
        self._model = None
        self._training_time: float = 0.0
        self._is_trained: bool = False

    @abstractmethod
    def build(self) -> None:
        """Build the underlying scikit-learn estimator.

        Must be implemented by subclasses to create and assign
        the specific model instance to self._model.
        """
        ...

    def train(self, X_train, y_train) -> None:
        """Train the model on the provided data.

        Tracks training time automatically.

        Args:
            X_train: Training feature matrix (sparse or dense).
            y_train: Training labels.

        Raises:
            RuntimeError: If build() was not called first.
        """
        if self._model is None:
            raise RuntimeError(
                f"{self.model_name}: build() must be called before train(). "
                f"Call build() first or use the model_factory."
            )

        if self._config.verbose:
            print(f"  Training {self.model_name}...", end=" ", flush=True)

        start = time.time()
        self._model.fit(X_train, y_train)
        self._training_time = time.time() - start
        self._is_trained = True

        if self._config.verbose:
            print(f"done ({self._training_time:.2f}s)")

    def predict(self, X) -> np.ndarray:
        """Generate predictions for the input data.

        Args:
            X: Feature matrix (sparse or dense).

        Returns:
            Array of predicted labels.

        Raises:
            RuntimeError: If model has not been trained.
        """
        self._check_trained()
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Generate probability predictions for the input data.

        Args:
            X: Feature matrix (sparse or dense).

        Returns:
            Array of shape (n_samples, n_classes) with probabilities.

        Raises:
            RuntimeError: If model has not been trained.
            AttributeError: If model does not support predict_proba.
        """
        self._check_trained()
        if not hasattr(self._model, "predict_proba"):
            raise AttributeError(
                f"{self.model_name} does not support predict_proba(). "
                f"Ensure probability=True is set for SVM."
            )
        return self._model.predict_proba(X)

    @property
    def model_name(self) -> str:
        """Return the human-readable model name."""
        return self.__class__.__name__

    @property
    def training_time(self) -> float:
        """Return training time in seconds."""
        return self._training_time

    @property
    def is_trained(self) -> bool:
        """Return whether the model has been trained."""
        return self._is_trained

    @property
    def sklearn_model(self) -> object:
        """Return the underlying scikit-learn estimator."""
        return self._model

    def get_metadata(self) -> dict:
        """Collect model metadata for reporting.

        Returns:
            Dictionary with model name, sklearn class, training
            time, trained status, and config.
        """
        return {
            "model_name": self.model_name,
            "sklearn_class": type(self._model).__name__ if self._model else None,
            "training_time_seconds": round(self._training_time, 4),
            "is_trained": self._is_trained,
            "random_state": self._config.random_state,
        }

    def _check_trained(self) -> None:
        """Verify model has been trained before prediction.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError(
                f"{self.model_name} has not been trained. "
                f"Call train() first."
            )
