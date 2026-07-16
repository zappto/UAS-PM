"""Persistence and visualization module for error analysis.

Saves CSVs, Markdown reports, and generates matplotlib figures.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from src.config.error_analysis_config import ErrorAnalysisConfig


def _plot_error_distribution(stats_df: pd.DataFrame, save_path: str, dpi: int):
    """Bar chart of correct vs incorrect predictions."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = stats_df.index
    correct = stats_df["correct_predictions"]
    errors = stats_df["errors"]
    
    ax.bar(models, correct, label='Correct', color='mediumseagreen')
    ax.bar(models, errors, bottom=correct, label='Errors', color='indianred')
    
    ax.set_ylabel('Number of Samples')
    ax.set_title('Prediction Distribution per Model')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _plot_top_confusions(confusions_df: pd.DataFrame, save_path: str, dpi: int):
    """Horizontal bar chart of top confused class pairs."""
    if confusions_df.empty:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_n = confusions_df.head(10).copy()
    top_n["pair"] = top_n["true_label"] + " \n→ " + top_n["predicted_label"]
    
    # Sort for horizontal bar plot
    top_n = top_n.sort_values("count", ascending=True)
    
    ax.barh(top_n["pair"], top_n["count"], color='salmon')
    
    ax.set_xlabel('Total Confusion Count')
    ax.set_title('Top Confused Class Pairs (All Models Aggregated)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def save_error_analysis_artifacts(
    stats_df: pd.DataFrame,
    difficult_classes_df: pd.DataFrame,
    top_confusions_df: pd.DataFrame,
    frequent_words_df: pd.DataFrame,
    misclassified_dfs: dict[str, pd.DataFrame],
    summary_md: str,
    config: ErrorAnalysisConfig | None = None,
) -> None:
    """Save all error analysis outputs to disk.

    Args:
        stats_df: Error statistics DataFrame.
        difficult_classes_df: DataFrame of classes ranked by difficulty.
        top_confusions_df: DataFrame of top confused pairs.
        frequent_words_df: DataFrame of frequent words in errors.
        misclassified_dfs: Dictionary of raw error DataFrames per model.
        summary_md: Markdown summary report content.
        config: Configuration.
    """
    if config is None:
        config = ErrorAnalysisConfig()

    base_dir = config.output_directory
    figures_dir = os.path.join(base_dir, "figures")
    raw_dir = os.path.join(base_dir, "raw")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Save Markdown summary
    with open(os.path.join(base_dir, "error_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    # 2. Save CSV Tables
    if config.save_tables:
        stats_df.to_csv(os.path.join(base_dir, "error_statistics.csv"))
        difficult_classes_df.to_csv(os.path.join(base_dir, "difficult_classes.csv"))
        top_confusions_df.to_csv(os.path.join(base_dir, "confusion_analysis.csv"), index=False)
        frequent_words_df.to_csv(os.path.join(base_dir, "frequent_error_words.csv"), index=False)
        
        # Save raw misclassified samples (combined or per model)
        # Here we save just one combined if requested, or the dictionary items.
        # Let's save them per model in the raw dir
        for model_name, df in misclassified_dfs.items():
            df.to_csv(os.path.join(raw_dir, f"{model_name}_misclassified.csv"), index=False)

    # 3. Save Figures
    if config.save_figures:
        fmt = config.figure_format
        dpi = config.dpi
        
        _plot_error_distribution(
            stats_df, 
            os.path.join(figures_dir, f"error_distribution.{fmt}"), 
            dpi
        )
        
        _plot_top_confusions(
            top_confusions_df, 
            os.path.join(figures_dir, f"top_confused_classes.{fmt}"), 
            dpi
        )

    print(f"  ✓ All error analysis artifacts saved to {base_dir}")
