"""Persistence module for evaluation artifacts.

Saves metrics, classification reports, confusion matrices,
and ROC curves to their respective directories.
"""

import json
import os
import pandas as pd

from src.config.evaluation_config import EvaluationConfig
from src.evaluation.classification_report import generate_markdown_report, report_to_dataframe
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.roc_curve import plot_roc_curve


def save_evaluation_artifacts(
    results: list[dict],
    comparison_df: pd.DataFrame,
    summary_md: str,
    config: EvaluationConfig | None = None,
) -> None:
    """Save all evaluation outputs to disk.

    Creates directory structure if it doesn't exist and delegates
    saving of individual artifacts.

    Args:
        results: List of raw evaluation results per model.
        comparison_df: Aggregated comparison metrics.
        summary_md: Markdown summary report content.
        config: Evaluation configuration.
    """
    if config is None:
        config = EvaluationConfig()

    base_dir = config.output_directory
    dirs = {
        "reports": os.path.join(base_dir, "classification_reports"),
        "cm": os.path.join(base_dir, "confusion_matrices"),
        "roc": os.path.join(base_dir, "roc_curves"),
        "comparison": os.path.join(base_dir, "comparison"),
    }

    # Ensure directories exist
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 1. Save main summary & comparison
    with open(os.path.join(base_dir, "evaluation_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)
    comparison_df.to_csv(os.path.join(base_dir, "model_comparison.csv"))
    print(f"  ✓ Saved summary and comparison CSV to {base_dir}")

    # Extract all metrics to JSON
    all_metrics = {}
    for r in results:
        all_metrics[r["model_name"]] = r["overall_metrics"]
    
    with open(os.path.join(base_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # 2. Save individual model artifacts
    for r in results:
        m_name = r["model_name"]
        safe_name = m_name.lower().replace(" ", "_")
        
        # Classification Report (CSV & MD)
        cr_df = report_to_dataframe(r["classification_report"])
        cr_df.to_csv(os.path.join(dirs["reports"], f"{safe_name}_report.csv"))
        
        cr_md = generate_markdown_report(r["classification_report"])
        with open(os.path.join(dirs["reports"], f"{safe_name}_report.md"), "w", encoding="utf-8") as f:
            f.write(f"# Classification Report: {m_name}\n\n{cr_md}")

        # Confusion Matrix (CSV)
        r["confusion_matrix_df"].to_csv(os.path.join(dirs["cm"], f"{safe_name}_cm.csv"))

        # Figures (CM & ROC)
        if config.save_figures:
            plot_confusion_matrix(
                cm=r["confusion_matrix_array"],
                labels=r["labels"],
                model_name=m_name,
                output_path=os.path.join(dirs["cm"], f"{safe_name}_cm.{config.figure_format}"),
                dpi=config.dpi,
            )

            if r.get("roc_data") is not None:
                plot_roc_curve(
                    roc_data=r["roc_data"],
                    labels=r["labels"],
                    model_name=m_name,
                    output_path=os.path.join(dirs["roc"], f"{safe_name}_roc.{config.figure_format}"),
                    dpi=config.dpi,
                )
    
    print("  ✓ All individual model artifacts saved.")
