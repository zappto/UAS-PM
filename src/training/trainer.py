"""Training orchestrator and report generator.

Provides functions to train individual models and all baseline
models. Generates training summary reports (markdown + CSV).
No evaluation — only training and saving.
"""

import os
from datetime import datetime

import pandas as pd

from src.config.model_config import ModelConfig
from src.training.base_model import BaseModel
from src.training.model_factory import create_model, get_available_models
from src.training.model_registry import get_model_save_path, get_model_info
from src.training.persistence import save_model


def train_model(
    model: BaseModel,
    X_train,
    y_train,
) -> BaseModel:
    """Train a single model. No evaluation.

    Args:
        model: Built (but untrained) model instance.
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        The trained model instance.
    """
    model.train(X_train, y_train)
    return model


def train_and_save_model(
    model_name: str,
    X_train,
    y_train,
    config: ModelConfig | None = None,
) -> BaseModel:
    """Create, train, and save a single baseline model.

    Args:
        model_name: Name of the model ('naive_bayes', etc.).
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Training configuration.

    Returns:
        The trained model instance.
    """
    if config is None:
        config = ModelConfig()

    # Create and build
    model = create_model(model_name, config)

    # Train
    train_model(model, X_train, y_train)

    # Save
    if config.save_model:
        filepath = get_model_save_path(model_name)
        save_model(model, filepath, overwrite=config.overwrite)

    return model


def train_all_models(
    X_train,
    y_train,
    config: ModelConfig | None = None,
) -> list[BaseModel]:
    """Train all registered baseline models.

    Trains Naive Bayes, Logistic Regression, and SVM
    sequentially with default parameters.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        config: Training configuration.

    Returns:
        List of trained BaseModel instances.
    """
    if config is None:
        config = ModelConfig()

    trained_models = []
    model_names = get_available_models()

    print("=" * 60)
    print("  Baseline Model Training")
    print("=" * 60)
    print(f"  Models: {model_names}")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Features: {X_train.shape[1]:,}")
    print()

    for i, name in enumerate(model_names, 1):
        print(f"[{i}/{len(model_names)}] {get_model_info(name)['display_name']}")
        model = train_and_save_model(name, X_train, y_train, config)
        trained_models.append(model)
        print()

    print("=" * 60)
    print("  ✓ All baseline models trained")
    print("=" * 60)

    return trained_models


def generate_training_report(
    trained_models: list[BaseModel],
    n_samples: int,
    n_features: int,
    config: ModelConfig | None = None,
) -> str:
    """Generate training summary as a markdown report.

    Contains only training metadata — no evaluation metrics.

    Args:
        trained_models: List of trained model instances.
        n_samples: Number of training samples.
        n_features: Number of features.
        config: Training configuration used.

    Returns:
        Complete markdown report as a string.
    """
    if config is None:
        config = ModelConfig()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = "# Baseline Model Training Summary\n\n"
    md += f"**Generated**: {timestamp}\n\n"
    md += "---\n\n"

    # Overview
    md += "## Overview\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    md += f"| Training Samples | {n_samples:,} |\n"
    md += f"| Number of Features | {n_features:,} |\n"
    md += f"| Models Trained | {len(trained_models)} |\n"
    md += f"| Random State | {config.random_state} |\n"
    md += f"| Parameter Mode | Default (Baseline) |\n"
    md += "\n---\n\n"

    # Per-model details
    md += "## Training Results\n\n"
    md += "| Model | Sklearn Class | Training Time (s) | Status |\n"
    md += "|-------|---------------|-------------------|--------|\n"
    for model in trained_models:
        meta = model.get_metadata()
        status = "✓ Trained" if meta["is_trained"] else "✗ Failed"
        md += (
            f"| {meta['model_name']} | {meta['sklearn_class']} | "
            f"{meta['training_time_seconds']:.4f} | {status} |\n"
        )

    md += "\n---\n\n"

    # Saved files
    md += "## Saved Model Files\n\n"
    md += "| Model | File |\n"
    md += "|-------|------|\n"
    for name in get_available_models():
        info = get_model_info(name)
        path = os.path.basename(info["save_path"])
        md += f"| {info['display_name']} | `models/{path}` |\n"

    md += "\n---\n\n"

    # Notes
    md += "## Notes\n\n"
    md += "- All models use **default parameters** (baseline).\n"
    md += "- No hyperparameter tuning was applied.\n"
    md += "- No evaluation metrics are included in this report.\n"
    md += "- Evaluation will be performed in `05_model_evaluation.ipynb`.\n"
    md += "- Hyperparameter tuning will be performed in `05_hyperparameter_tuning.ipynb`.\n"

    md += "\n---\n\n"
    md += "*Report generated automatically by the training pipeline.*\n"

    return md


def export_training_statistics(
    trained_models: list[BaseModel],
    n_samples: int,
    n_features: int,
    filepath: str,
) -> None:
    """Export training statistics as CSV.

    Args:
        trained_models: List of trained model instances.
        n_samples: Number of training samples.
        n_features: Number of features.
        filepath: Destination CSV path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rows = []
    for model in trained_models:
        meta = model.get_metadata()
        rows.append({
            "model_name": meta["model_name"],
            "sklearn_class": meta["sklearn_class"],
            "training_samples": n_samples,
            "n_features": n_features,
            "training_time_seconds": meta["training_time_seconds"],
            "is_trained": meta["is_trained"],
            "random_state": meta["random_state"],
        })

    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"  ✓ Training statistics exported: {filepath}")


def save_training_report(content: str, filepath: str) -> None:
    """Save training report to file.

    Args:
        content: Markdown report string.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ Training report saved: {filepath}")
