#!/usr/bin/env python3
"""
EDA Runner Script
=================
Executes the complete Exploratory Data Analysis pipeline.
This script mirrors the notebook logic and can be run standalone.

Usage:
    cd /home/zapp/Kampus/PM
    python3 -m src.eda_runner
"""

import os
import sys

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config.settings import REPORTS_DIR, FIGURES_DIR
from src.data.load_dataset import load_dataset
from src.data.dataset_summary import (
    get_dataset_overview,
    get_missing_values,
    get_duplicate_count,
)
from src.data.text_analysis import (
    compute_text_lengths,
    get_text_statistics,
    get_vocabulary_stats,
    get_top_words,
    get_top_ngrams,
)
from src.data.label_analysis import (
    get_label_distribution,
    get_class_count,
    get_class_imbalance_ratio,
)
from src.visualization.plots import (
    plot_label_distribution,
    plot_label_distribution_pie,
    plot_text_length_histogram,
    plot_text_length_boxplot,
    plot_top_words,
    plot_top_ngrams,
    plot_missing_values,
    plot_duplicate_summary,
)
from src.utils.report_generator import (
    assess_data_quality,
    generate_eda_report,
    save_report,
    export_dataset_overview,
    export_label_distribution,
    export_text_statistics,
    export_vocabulary_statistics,
)


def main() -> None:
    """Execute the full EDA pipeline."""
    # Ensure output directories exist
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("  EXPLORATORY DATA ANALYSIS — Cyberbullying Classification")
    print("=" * 70)

    # ── 1. Load Dataset ──────────────────────────────────────────────────
    print("\n[1/8] Loading dataset...")
    df = load_dataset()

    # ── 2. Dataset Overview ──────────────────────────────────────────────
    print("\n[2/8] Computing dataset overview...")
    overview = get_dataset_overview(df)
    missing_df = get_missing_values(df)
    duplicate_info = get_duplicate_count(df)

    print(f"  Shape: {overview['shape']}")
    print(f"  Memory: {overview['memory_usage']['total_mb']} MB")
    print(f"  Duplicates: {duplicate_info['duplicate_count']}")

    # ── 3. Label Analysis ────────────────────────────────────────────────
    print("\n[3/8] Analyzing labels...")
    label_dist = get_label_distribution(df)
    class_count = get_class_count(df)
    imbalance = get_class_imbalance_ratio(df)

    print(f"  Classes: {class_count}")
    print(f"  Imbalance: {imbalance['status']} ({imbalance['ratio']}:1)")
    for _, row in label_dist.iterrows():
        print(f"    {row['label']}: {int(row['count']):,} ({row['percentage']}%)")

    # ── 4. Text Analysis ─────────────────────────────────────────────────
    print("\n[4/8] Analyzing text characteristics...")
    df_with_lengths = compute_text_lengths(df)
    text_stats = get_text_statistics(df_with_lengths)
    vocab_stats = get_vocabulary_stats(df)

    wc = text_stats["word_count"]
    print(f"  Word count — Mean: {wc['mean']}, Median: {wc['median']}, "
          f"Std: {wc['std']}, Min: {wc['min']}, Max: {wc['max']}")
    print(f"  Vocabulary: {vocab_stats['unique_words']:,} unique words")

    # ── 5. Word & N-gram Analysis ────────────────────────────────────────
    print("\n[5/8] Computing word frequencies and n-grams...")
    top_20_words = get_top_words(df, n=20)
    top_50_words = get_top_words(df, n=50)
    top_bigrams = get_top_ngrams(df, n=2, top_k=20)
    top_trigrams = get_top_ngrams(df, n=3, top_k=20)

    print(f"  Top 5 words: {[w for w, _ in top_20_words[:5]]}")
    print(f"  Top 5 bigrams: {[b for b, _ in top_bigrams[:5]]}")

    # ── 6. Data Quality ──────────────────────────────────────────────────
    print("\n[6/8] Assessing data quality...")
    quality_issues = assess_data_quality(df)
    for issue, count in quality_issues.items():
        status = "✓" if count == 0 else f"⚠ {count:,}"
        print(f"  {issue}: {status}")

    # ── 7. Generate Visualizations ───────────────────────────────────────
    print("\n[7/8] Generating visualizations...")

    figures = {
        "label_distribution.png": plot_label_distribution(label_dist),
        "label_distribution_pie.png": plot_label_distribution_pie(label_dist),
        "text_length_histogram.png": plot_text_length_histogram(df_with_lengths),
        "text_length_boxplot.png": plot_text_length_boxplot(df_with_lengths),
        "top_words.png": plot_top_words(top_20_words),
        "top_bigrams.png": plot_top_ngrams(top_bigrams, title="Top 20 Bigrams"),
        "top_trigrams.png": plot_top_ngrams(top_trigrams, title="Top 20 Trigrams"),
        "missing_values.png": plot_missing_values(missing_df),
        "duplicate_summary.png": plot_duplicate_summary(duplicate_info),
    }

    for name, fig in figures.items():
        path = os.path.join(FIGURES_DIR, name)
        fig.savefig(path, bbox_inches="tight")
        print(f"  ✓ Saved: {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ── 8. Generate Reports ──────────────────────────────────────────────
    print("\n[8/8] Generating reports...")

    # Markdown report
    report_content = generate_eda_report(
        overview=overview,
        label_dist=label_dist,
        text_stats=text_stats,
        vocab_stats=vocab_stats,
        imbalance=imbalance,
        duplicate_info=duplicate_info,
        missing_df=missing_df,
        quality_issues=quality_issues,
    )
    save_report(report_content, os.path.join(REPORTS_DIR, "eda_summary.md"))

    # CSV exports
    export_dataset_overview(overview, os.path.join(REPORTS_DIR, "dataset_overview.csv"))
    export_label_distribution(label_dist, os.path.join(REPORTS_DIR, "label_distribution.csv"))
    export_text_statistics(text_stats, os.path.join(REPORTS_DIR, "text_statistics.csv"))
    export_vocabulary_statistics(vocab_stats, os.path.join(REPORTS_DIR, "vocabulary_statistics.csv"))

    print("\n" + "=" * 70)
    print("  EDA COMPLETE")
    print("=" * 70)
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
