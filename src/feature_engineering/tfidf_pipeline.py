"""TF-IDF feature engineering pipeline orchestrator.

Executes the complete pipeline:
    Load → Split → Fit TF-IDF → Transform → Statistics → Report

All steps use the modular components from the feature_engineering
package. The vectorizer is fit only on training data to prevent
data leakage.
"""

import os
from datetime import datetime

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.tfidf_config import TfidfConfig
from src.feature_engineering.vectorizer import create_tfidf_vectorizer
from src.feature_engineering.train_test_split import split_dataset
from src.feature_engineering.feature_statistics import (
    compute_feature_statistics,
    get_top_tfidf_terms,
)


def run_tfidf_pipeline(
    df: pd.DataFrame,
    config: TfidfConfig | None = None,
    text_col: str = "text_clean",
    label_col: str = "label",
) -> tuple[spmatrix, spmatrix, pd.Series, pd.Series, TfidfVectorizer, dict]:
    """Execute the full TF-IDF feature engineering pipeline.

    Pipeline steps:
        1. Split dataset (stratified train/test)
        2. Create TF-IDF vectorizer from config
        3. Fit vectorizer on X_train only (prevent data leakage)
        4. Transform X_train and X_test
        5. Compute feature statistics

    Args:
        df: Input DataFrame with text and label columns.
        config: TF-IDF configuration. Uses defaults if None.
        text_col: Name of the preprocessed text column.
        label_col: Name of the label column.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, vectorizer, stats):
            - X_train: Sparse TF-IDF matrix for training.
            - X_test: Sparse TF-IDF matrix for testing.
            - y_train: Training labels (pd.Series).
            - y_test: Testing labels (pd.Series).
            - vectorizer: Fitted TfidfVectorizer.
            - stats: Feature statistics dictionary.

    Raises:
        ValueError: If required columns are missing or dataset is empty.
    """
    if config is None:
        config = TfidfConfig()

    print("=" * 60)
    print("  TF-IDF Feature Engineering Pipeline")
    print("=" * 60)

    # Step 1: Split dataset
    print("\n[1/4] Splitting dataset...")
    X_train_text, X_test_text, y_train, y_test = split_dataset(
        df=df,
        text_col=text_col,
        label_col=label_col,
        config=config,
    )

    # Step 2: Create vectorizer
    print("\n[2/4] Creating TF-IDF vectorizer...")
    vectorizer = create_tfidf_vectorizer(config)
    params = config.get_vectorizer_params()
    print(f"  Config: max_features={params['max_features']}, "
          f"min_df={params['min_df']}, max_df={params['max_df']}, "
          f"ngram_range={params['ngram_range']}")

    # Step 3: Fit on training data ONLY (prevent data leakage)
    print("\n[3/4] Fitting and transforming...")
    X_train = vectorizer.fit_transform(X_train_text)
    print(f"  ✓ Fitted on X_train: {X_train.shape}")

    # Transform test data with the same fitted vectorizer
    X_test = vectorizer.transform(X_test_text)
    print(f"  ✓ Transformed X_test: {X_test.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Step 4: Compute statistics
    print("\n[4/4] Computing statistics...")
    stats = compute_feature_statistics(
        X_train, X_test, vectorizer, y_train, y_test,
    )
    print(f"  Matrix density: {stats['matrix_density']:.6f}")
    print(f"  Matrix sparsity: {stats['matrix_sparsity']:.6f}")
    print(f"  Avg features/doc: {stats['avg_features_per_document']:.1f}")

    print("\n" + "=" * 60)
    print("  ✓ TF-IDF Pipeline Complete")
    print("=" * 60)

    return X_train, X_test, y_train, y_test, vectorizer, stats


def generate_feature_report(
    stats: dict,
    config: TfidfConfig,
    validation: dict,
    top_terms: list[tuple[str, float]] | None = None,
) -> str:
    """Generate feature engineering summary as a markdown report.

    Args:
        stats: Feature statistics dictionary.
        config: TF-IDF configuration used.
        validation: Validation results dictionary.
        top_terms: Optional list of top TF-IDF terms.

    Returns:
        Complete markdown report as a string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = config.to_dict()

    md = "# Feature Engineering Report\n\n"
    md += f"**Generated**: {timestamp}\n\n"
    md += "---\n\n"

    # Dataset
    md += "## 1. Dataset\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    md += f"| Input File | `dataset/processed/final_dataset_clean.csv` |\n"
    md += f"| Text Column | `text_clean` |\n"
    md += f"| Training Samples | {stats['train_samples']:,} |\n"
    md += f"| Testing Samples | {stats['test_samples']:,} |\n"
    md += f"| Total Samples | {stats['train_samples'] + stats['test_samples']:,} |\n"
    md += "\n---\n\n"

    # Split
    md += "## 2. Train/Test Split\n\n"
    md += "| Parameter | Value |\n"
    md += "|-----------|-------|\n"
    md += f"| Test Size | {cfg['test_size']} |\n"
    md += f"| Random State | {cfg['random_state']} |\n"
    md += f"| Stratified | {cfg['stratify']} |\n"
    md += "\n### Training Label Distribution\n\n"
    md += "| Label | Count |\n"
    md += "|-------|-------|\n"
    for label, count in sorted(
        stats["train_label_distribution"].items(),
        key=lambda x: x[1], reverse=True,
    ):
        md += f"| `{label}` | {count:,} |\n"
    md += "\n### Testing Label Distribution\n\n"
    md += "| Label | Count |\n"
    md += "|-------|-------|\n"
    for label, count in sorted(
        stats["test_label_distribution"].items(),
        key=lambda x: x[1], reverse=True,
    ):
        md += f"| `{label}` | {count:,} |\n"
    md += "\n---\n\n"

    # TF-IDF Config
    md += "## 3. TF-IDF Configuration\n\n"
    md += "| Parameter | Value |\n"
    md += "|-----------|-------|\n"
    for param, value in cfg.items():
        if param not in ("test_size", "random_state", "stratify"):
            md += f"| {param} | `{value}` |\n"
    md += "\n---\n\n"

    # Feature Statistics
    md += "## 4. Feature Statistics\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    md += f"| Vocabulary Size | {stats['vocabulary_size']:,} |\n"
    md += f"| Number of Features | {stats['n_features']:,} |\n"
    md += f"| Training Matrix Shape | {stats['train_shape']} |\n"
    md += f"| Testing Matrix Shape | {stats['test_shape']} |\n"
    md += f"| Matrix Density | {stats['matrix_density']:.6f} |\n"
    md += f"| Matrix Sparsity | {stats['matrix_sparsity']:.6f} |\n"
    md += f"| Avg Features per Document | {stats['avg_features_per_document']:.2f} |\n"
    md += f"| Min Features per Document | {stats['min_features_per_document']} |\n"
    md += f"| Max Features per Document | {stats['max_features_per_document']} |\n"
    md += "\n---\n\n"

    # Top Terms
    if top_terms:
        md += "## 5. Top TF-IDF Terms\n\n"
        md += "| Rank | Term | Avg TF-IDF |\n"
        md += "|------|------|------------|\n"
        for i, (term, score) in enumerate(top_terms[:20], 1):
            md += f"| {i} | `{term}` | {score:.6f} |\n"
        md += "\n---\n\n"

    # Validation
    section_num = 6 if top_terms else 5
    md += f"## {section_num}. Validation\n\n"
    md += f"**Status**: {'✓ PASS' if validation['is_valid'] else '✗ FAILED'}\n\n"
    md += "| Check | Result |\n"
    md += "|-------|--------|\n"
    for check, passed in validation["checks"].items():
        md += f"| {check} | {'✓' if passed else '✗'} |\n"
    md += "\n---\n\n"

    # Recommendations
    section_num += 1
    md += f"## {section_num}. Recommendations for Model Training\n\n"
    md += "1. Use the saved `tfidf_vectorizer.joblib` to transform new text.\n"
    md += "2. Use stratified split to maintain class distribution.\n"
    md += "3. Evaluate using F1-Score (macro/weighted) as primary metric.\n"
    md += f"4. Feature dimensionality is {stats['n_features']:,} — "
    md += "manageable for NB, LR, and SVM.\n"
    md += f"5. Matrix sparsity is {stats['matrix_sparsity']:.4f} — "
    md += "sparse-aware algorithms (SVM, LR) should perform well.\n"
    md += "6. Consider class weights for handling class imbalance.\n"
    md += "\n---\n\n"
    md += "*Report generated automatically by the feature engineering pipeline.*\n"

    return md


def save_feature_report(content: str, filepath: str) -> None:
    """Save feature report to file.

    Args:
        content: Markdown string.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ Report saved: {filepath}")
