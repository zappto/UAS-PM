"""Artifact persistence for TF-IDF pipeline.

Save and load TF-IDF vectorizer, sparse feature matrices,
and label series using joblib, scipy, and pandas.
"""

import os

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import (
    TFIDF_VECTORIZER_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    MODELS_DIR,
)


def save_tfidf_artifacts(
    vectorizer: TfidfVectorizer,
    X_train,
    X_test,
    y_train: pd.Series,
    y_test: pd.Series,
    overwrite: bool = False,
) -> dict[str, str]:
    """Save all TF-IDF pipeline artifacts to disk.

    Saves:
        - Vectorizer → joblib (.joblib)
        - Feature matrices → scipy sparse (.npz)
        - Labels → pandas CSV (.csv)

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        X_train: Training feature matrix (sparse).
        X_test: Testing feature matrix (sparse).
        y_train: Training labels.
        y_test: Testing labels.
        overwrite: Allow overwriting existing files.

    Returns:
        Dictionary mapping artifact names to saved file paths.

    Raises:
        FileExistsError: If any file exists and overwrite is False.
    """
    paths = {
        "vectorizer": TFIDF_VECTORIZER_PATH,
        "X_train": X_TRAIN_PATH,
        "X_test": X_TEST_PATH,
        "y_train": Y_TRAIN_PATH,
        "y_test": Y_TEST_PATH,
    }

    # Check for existing files
    if not overwrite:
        existing = [p for p in paths.values() if os.path.exists(p)]
        if existing:
            raise FileExistsError(
                f"Files already exist: {existing}. "
                f"Set overwrite=True to replace."
            )

    # Create directories
    for filepath in paths.values():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save vectorizer
    joblib.dump(vectorizer, paths["vectorizer"])
    print(f"  ✓ Vectorizer saved: {paths['vectorizer']}")

    # Save sparse matrices
    sparse.save_npz(paths["X_train"], X_train)
    print(f"  ✓ X_train saved: {paths['X_train']} "
          f"(shape={X_train.shape})")

    sparse.save_npz(paths["X_test"], X_test)
    print(f"  ✓ X_test saved: {paths['X_test']} "
          f"(shape={X_test.shape})")

    # Save labels
    y_train.to_csv(paths["y_train"], index=False, header=True)
    print(f"  ✓ y_train saved: {paths['y_train']} "
          f"({len(y_train):,} rows)")

    y_test.to_csv(paths["y_test"], index=False, header=True)
    print(f"  ✓ y_test saved: {paths['y_test']} "
          f"({len(y_test):,} rows)")

    return paths


def load_tfidf_artifacts() -> tuple[
    TfidfVectorizer, object, object, pd.Series, pd.Series
]:
    """Load all saved TF-IDF artifacts from disk.

    Returns:
        Tuple of (vectorizer, X_train, X_test, y_train, y_test).

    Raises:
        FileNotFoundError: If any required artifact file is missing.
    """
    required = {
        "vectorizer": TFIDF_VECTORIZER_PATH,
        "X_train": X_TRAIN_PATH,
        "X_test": X_TEST_PATH,
        "y_train": Y_TRAIN_PATH,
        "y_test": Y_TEST_PATH,
    }

    # Check all files exist
    missing = [name for name, path in required.items()
               if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact files: {missing}. "
            f"Run the feature engineering pipeline first."
        )

    vectorizer = joblib.load(required["vectorizer"])
    X_train = sparse.load_npz(required["X_train"])
    X_test = sparse.load_npz(required["X_test"])
    y_train = pd.read_csv(required["y_train"]).squeeze("columns")
    y_test = pd.read_csv(required["y_test"]).squeeze("columns")

    print(f"✓ Artifacts loaded: X_train={X_train.shape}, "
          f"X_test={X_test.shape}, "
          f"vocab={len(vectorizer.vocabulary_):,}")

    return vectorizer, X_train, X_test, y_train, y_test
