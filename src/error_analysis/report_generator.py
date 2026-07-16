"""Report generator for error analysis.

Generates the final error_summary.md file.
"""

import pandas as pd
from datetime import datetime


def generate_error_summary(
    stats_df: pd.DataFrame,
    difficult_classes_df: pd.DataFrame,
    top_confusions_df: pd.DataFrame,
) -> str:
    """Generate Markdown summary of the error analysis.

    Args:
        stats_df: Error statistics DataFrame.
        difficult_classes_df: DataFrame of classes ranked by recall error rate.
        top_confusions_df: DataFrame of top confused pairs.

    Returns:
        Markdown string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models = list(stats_df.index)
    total_samples = stats_df.iloc[0]["total_samples"] if not stats_df.empty else 0

    md = "# Error Analysis Summary\n\n"
    md += f"**Analysis Date**: {timestamp}\n\n"
    md += "---\n\n"

    # Overview
    md += "## Overview\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    md += f"| Models Analyzed | {', '.join(models)} |\n"
    md += f"| Dataset Used | `X_test.npz`, `y_test.csv`, `final_dataset_clean.csv` |\n"
    md += f"| Test Samples | {total_samples:,} |\n"
    md += "\n---\n\n"

    # Error Statistics
    md += "## Error Statistics\n\n"
    
    # Format percentages for display
    display_stats = stats_df.copy()
    display_stats["error_percentage"] = display_stats["error_percentage"].apply(lambda x: f"{x:.2f}%")
    display_stats["correct_percentage"] = display_stats["correct_percentage"].apply(lambda x: f"{x:.2f}%")
    
    md += display_stats.to_markdown()
    md += "\n\n---\n\n"

    # Difficult Classes
    md += "## Most Difficult Classes\n\n"
    md += "Ranked by average Recall Error Rate (False Negative Rate) across all models.\n\n"
    
    if not difficult_classes_df.empty:
        # Show top 3
        top_difficult = difficult_classes_df.head(3)
        display_diff = top_difficult.copy()
        for col in display_diff.columns:
            if "rate" in col:
                display_diff[col] = display_diff[col].apply(lambda x: f"{x:.4f}")
        md += display_diff.to_markdown()
    else:
        md += "*No error data available.*"
    md += "\n\n---\n\n"

    # Confused Classes
    md += "## Most Frequently Confused Classes\n\n"
    md += "Aggregated top pairwise misclassifications across all models.\n\n"
    
    if not top_confusions_df.empty:
        md += top_confusions_df.head(10).to_markdown(index=False)
    else:
        md += "*No confusion data available.*"
    md += "\n\n---\n\n"

    # File References
    md += "## Generated Files\n\n"
    md += "- **Statistics**: `error_statistics.csv`\n"
    md += "- **Class Errors**: `class_error_analysis.csv`, `difficult_classes.csv`\n"
    md += "- **Confusions**: `confusion_analysis.csv`\n"
    md += "- **Text Patterns**: `frequent_error_words.csv`\n"
    md += "- **Raw Data**: `raw/misclassified_samples.csv`\n"
    md += "- **Figures**: See `figures/` directory.\n"
    
    md += "\n---\n\n"
    md += "*Report generated automatically by the error analysis pipeline.*\n"

    return md
