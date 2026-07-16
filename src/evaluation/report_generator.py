"""Evaluation report generator.

Generates the final evaluation_summary.md report.
"""

import os
from datetime import datetime
import pandas as pd


def generate_evaluation_summary(
    comparison_df: pd.DataFrame,
    results: list[dict],
    n_samples: int,
) -> str:
    """Generate the main evaluation summary markdown report.

    Contains no interpretation or conclusions, just raw results
    and links to generated artifacts.

    Args:
        comparison_df: Aggregated comparison metrics.
        results: List of raw evaluation results per model.
        n_samples: Number of samples in the test set.

    Returns:
        Markdown string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = "# Model Evaluation Summary\n\n"
    md += f"**Evaluation Date**: {timestamp}\n\n"
    md += "---\n\n"

    # Overview
    md += "## Overview\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    md += f"| Models Evaluated | {len(results)} |\n"
    md += f"| Dataset Used | `X_test.npz`, `y_test.csv` |\n"
    md += f"| Test Samples | {n_samples:,} |\n"
    md += f"| Primary Metric | F1 Score (Macro) |\n"
    md += "\n---\n\n"

    # Comparison Table
    md += "## Overall Model Comparison\n\n"
    
    # Format DataFrame for Markdown
    display_df = comparison_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    md += display_df.to_markdown()
    md += "\n\n"
    md += "[📥 Download full comparison CSV](model_comparison.csv)\n\n"
    md += "---\n\n"

    # Visualizations Links
    md += "## Evaluation Visualizations\n\n"
    
    for r in results:
        m = r["model_name"]
        
        md += f"### {m}\n\n"
        
        # Classification report
        safe_name = m.lower().replace(" ", "_")
        md += f"- **Classification Report**: [`{safe_name}_report.csv`](classification_reports/{safe_name}_report.csv)\n"
        
        # Confusion Matrix
        md += f"- **Confusion Matrix**: [`{safe_name}_cm.png`](confusion_matrices/{safe_name}_cm.png)\n"
        
        # ROC Curve
        if r.get("roc_data") is not None:
            md += f"- **ROC Curve (OvR)**: [`{safe_name}_roc.png`](roc_curves/{safe_name}_roc.png)\n"
        
        md += "\n"

    md += "---\n\n"

    # Notes
    md += "## Notes\n\n"
    md += "- All metrics calculated on the isolated **testing dataset**.\n"
    md += "- ROC Curves use the **One-vs-Rest (OvR)** approach for multi-class classification.\n"
    md += "- This report contains objective metrics only. See `07_error_analysis.ipynb` for detailed analysis.\n"
    md += "\n---\n\n"
    md += "*Report generated automatically by the evaluation pipeline.*\n"

    return md
