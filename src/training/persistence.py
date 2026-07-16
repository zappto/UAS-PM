"""Model persistence — save and load trained models.

Uses joblib for serialization, which handles scikit-learn
estimators efficiently.
"""

import os

import joblib

from src.training.base_model import BaseModel


def save_model(
    model: BaseModel,
    filepath: str,
    overwrite: bool = False,
) -> str:
    """Save a trained model to disk using joblib.

    Saves the underlying scikit-learn estimator, not the
    wrapper class.

    Args:
        model: Trained BaseModel instance.
        filepath: Destination .joblib file path.
        overwrite: Allow overwriting existing files.

    Returns:
        The filepath where the model was saved.

    Raises:
        FileExistsError: If file exists and overwrite is False.
        RuntimeError: If model has not been trained.
    """
    if not model.is_trained:
        raise RuntimeError(
            f"Cannot save untrained model: {model.model_name}. "
            f"Train the model first."
        )

    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"Model file already exists: {filepath}. "
            f"Set overwrite=True to replace it."
        )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model.sklearn_model, filepath)

    file_size = os.path.getsize(filepath)
    size_str = _format_size(file_size)
    print(f"  ✓ Model saved: {filepath} ({size_str})")

    return filepath


def load_model(filepath: str) -> object:
    """Load a trained scikit-learn model from a joblib file.

    Args:
        filepath: Path to the .joblib file.

    Returns:
        The loaded scikit-learn estimator.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model file not found: {filepath}. "
            f"Train the model first."
        )

    model = joblib.load(filepath)
    print(f"✓ Model loaded: {filepath} ({type(model).__name__})")
    return model


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Formatted string (e.g., '1.2 MB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
