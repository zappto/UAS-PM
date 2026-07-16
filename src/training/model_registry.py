"""Model registry — metadata for all registered models.

Provides model metadata including save paths, descriptions,
and sklearn class names. Designed for easy expansion with
future models.
"""

from src.config.settings import (
    NB_MODEL_PATH,
    LR_MODEL_PATH,
    SVM_MODEL_PATH,
)


AVAILABLE_MODELS: dict[str, dict] = {
    "naive_bayes": {
        "display_name": "Naive Bayes",
        "sklearn_class": "MultinomialNB",
        "save_path": NB_MODEL_PATH,
        "description": "Multinomial Naive Bayes — Baseline classifier for NLP",
    },
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "sklearn_class": "LogisticRegression",
        "save_path": LR_MODEL_PATH,
        "description": "Logistic Regression — Linear probabilistic classifier",
    },
    "svm": {
        "display_name": "Support Vector Machine",
        "sklearn_class": "SVC",
        "save_path": SVM_MODEL_PATH,
        "description": "Linear SVM — High-performance sparse text classifier",
    },
}


def get_all_model_names() -> list[str]:
    """Return all registered model names.

    Returns:
        List of model name strings.
    """
    return list(AVAILABLE_MODELS.keys())


def get_model_save_path(model_name: str) -> str:
    """Return the default save path for a model.

    Args:
        model_name: Registered model name.

    Returns:
        Absolute file path for saving the model.

    Raises:
        ValueError: If model_name is not registered.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {get_all_model_names()}"
        )
    return AVAILABLE_MODELS[model_name]["save_path"]


def get_model_info(model_name: str) -> dict:
    """Return full metadata for a model.

    Args:
        model_name: Registered model name.

    Returns:
        Dictionary with display_name, sklearn_class, save_path,
        and description.

    Raises:
        ValueError: If model_name is not registered.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {get_all_model_names()}"
        )
    return AVAILABLE_MODELS[model_name].copy()
