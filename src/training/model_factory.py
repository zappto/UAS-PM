"""Model factory — instantiate models by name.

Provides a centralized factory function to create model
instances without duplicating initialization logic.
"""

from src.config.model_config import ModelConfig
from src.training.base_model import BaseModel
from src.training.naive_bayes import NaiveBayesModel
from src.training.logistic_regression import LogisticRegressionModel
from src.training.svm import SVMModel


_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "naive_bayes": NaiveBayesModel,
    "logistic_regression": LogisticRegressionModel,
    "svm": SVMModel,
}


def create_model(
    model_name: str,
    config: ModelConfig | None = None,
) -> BaseModel:
    """Create a model instance by name.

    The model is built (initialized) but not trained.

    Args:
        model_name: Name of the model to create. Must be one of:
            'naive_bayes', 'logistic_regression', 'svm'.
        config: Training configuration.

    Returns:
        A built (but untrained) BaseModel instance.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available}"
        )

    model_class = _MODEL_REGISTRY[model_name]
    model = model_class(config)
    model.build()
    return model


def get_available_models() -> list[str]:
    """Return list of available model names.

    Returns:
        List of registered model name strings.
    """
    return list(_MODEL_REGISTRY.keys())
