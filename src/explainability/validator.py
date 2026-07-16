"""Validation module for explainability components.

This module provides functions to validate models, vectorizers,
and predictions before running explanations.
"""

from typing import Any
import numpy as np


class ValidationError(Exception):
    """Custom exception for validation errors in explainability."""
    pass


def validate_model(model: Any) -> None:
    """Validate that the model has required attributes for explanation.

    Args:
        model: The trained ML model.
    Raises:
        ValidationError: If the model is not supported or missing attributes.
    """
    if model is None:
        raise ValidationError("Model is None.")
    
    if not hasattr(model, "predict"):
        raise ValidationError("Model missing 'predict' method.")
    
    # Check if it's a known model type (LR, SVM, NB)
    model_type = type(model).__name__
    supported_models = ["LogisticRegression", "LinearSVC", "SVC", "MultinomialNB"]
    if model_type not in supported_models:
        raise ValidationError(f"Model type '{model_type}' is not supported for explainability.")


def validate_vectorizer(vectorizer: Any) -> None:
    """Validate TF-IDF vectorizer.

    Args:
        vectorizer: The TF-IDF vectorizer.
    Raises:
        ValidationError: If the vectorizer is invalid.
    """
    if vectorizer is None:
        raise ValidationError("Vectorizer is None.")
    
    if not hasattr(vectorizer, "get_feature_names_out"):
        raise ValidationError("Vectorizer missing 'get_feature_names_out' method.")


def validate_feature_names(feature_names: np.ndarray, num_features: int) -> None:
    """Validate that feature names match the expected number of features.

    Args:
        feature_names: Array of feature names.
        num_features: Expected number of features.
    Raises:
        ValidationError: If the counts do not match.
    """
    if len(feature_names) != num_features:
        raise ValidationError(
            f"Feature names count ({len(feature_names)}) does not match "
            f"model features count ({num_features})."
        )


def validate_predictions(predictions: np.ndarray, x_data: Any) -> None:
    """Validate prediction shape against input data shape.

    Args:
        predictions: Array of predictions.
        x_data: Input data array or sparse matrix.
    Raises:
        ValidationError: If shapes are incompatible.
    """
    if predictions is None:
        raise ValidationError("Predictions array is None.")
    
    if len(predictions) != x_data.shape[0]:
        raise ValidationError(
            f"Predictions length ({len(predictions)}) does not match "
            f"input data length ({x_data.shape[0]})."
        )
