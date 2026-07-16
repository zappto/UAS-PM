"""Feature statistics for TF-IDF matrices.

Computes vocabulary size, matrix shape, density, sparsity,
top TF-IDF terms, and per-document feature statistics.
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_feature_statistics(
    X_train,
    X_test,
    vectorizer: TfidfVectorizer,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Compute comprehensive statistics for TF-IDF feature matrices.

    Args:
        X_train: Training feature matrix (sparse).
        X_test: Testing feature matrix (sparse).
        vectorizer: Fitted TF-IDF vectorizer.
        y_train: Training labels.
        y_test: Testing labels.

    Returns:
        Dictionary with all computed statistics.
    """
    vocab = vectorizer.vocabulary_
    vocab_size = len(vocab)
    n_features = X_train.shape[1]

    # Matrix density / sparsity
    train_nnz = X_train.nnz if issparse(X_train) else np.count_nonzero(X_train)
    train_total = X_train.shape[0] * X_train.shape[1]
    density = train_nnz / train_total if train_total > 0 else 0.0
    sparsity = 1.0 - density

    # Average non-zero features per document
    if issparse(X_train):
        nnz_per_doc = np.diff(X_train.tocsr().indptr)
    else:
        nnz_per_doc = np.count_nonzero(X_train, axis=1)
    avg_features_per_doc = float(np.mean(nnz_per_doc))

    return {
        "vocabulary_size": vocab_size,
        "n_features": n_features,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "train_nnz": int(train_nnz),
        "matrix_density": round(density, 6),
        "matrix_sparsity": round(sparsity, 6),
        "avg_features_per_document": round(avg_features_per_doc, 2),
        "min_features_per_document": int(np.min(nnz_per_doc)),
        "max_features_per_document": int(np.max(nnz_per_doc)),
        "train_labels": int(y_train.nunique()),
        "test_labels": int(y_test.nunique()),
        "train_label_distribution": y_train.value_counts().to_dict(),
        "test_label_distribution": y_test.value_counts().to_dict(),
    }


def get_top_tfidf_terms(
    vectorizer: TfidfVectorizer,
    X_train,
    n: int = 30,
) -> list[tuple[str, float]]:
    """Get the top N terms by average TF-IDF score across training corpus.

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        X_train: Training feature matrix (sparse).
        n: Number of top terms to return.

    Returns:
        List of (term, avg_tfidf_score) tuples sorted by score descending.
    """
    feature_names = vectorizer.get_feature_names_out()

    # Compute mean TF-IDF score per term across all training docs
    if issparse(X_train):
        mean_tfidf = np.array(X_train.mean(axis=0)).flatten()
    else:
        mean_tfidf = np.mean(X_train, axis=0)

    # Get top N indices
    top_indices = np.argsort(mean_tfidf)[::-1][:n]

    return [
        (str(feature_names[i]), round(float(mean_tfidf[i]), 6))
        for i in top_indices
    ]


def export_feature_statistics(stats: dict, filepath: str) -> None:
    """Export feature statistics as a CSV file.

    Args:
        stats: Statistics dictionary from compute_feature_statistics().
        filepath: Destination CSV path.
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rows = []
    for key, value in stats.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                rows.append({"metric": f"{key}_{sub_key}", "value": sub_val})
        elif isinstance(value, tuple):
            rows.append({"metric": key, "value": str(value)})
        else:
            rows.append({"metric": key, "value": value})

    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"  ✓ Statistics exported: {filepath}")


def export_feature_importance_preview(
    top_terms: list[tuple[str, float]],
    filepath: str,
) -> None:
    """Export top TF-IDF terms preview as CSV.

    Args:
        top_terms: List of (term, avg_score) tuples.
        filepath: Destination CSV path.
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = pd.DataFrame(top_terms, columns=["term", "avg_tfidf_score"])
    df["rank"] = range(1, len(df) + 1)
    df = df[["rank", "term", "avg_tfidf_score"]]
    df.to_csv(filepath, index=False)
    print(f"  ✓ Feature importance preview exported: {filepath}")
