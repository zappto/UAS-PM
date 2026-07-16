"""Tuning report generator.

Generates tuning summary reports in markdown and CSV format.
Reports include parameter grids, best parameters, CV scores,
search duration, and candidate counts. No evaluation metrics.
"""

import os
from datetime import datetime

import pandas as pd


def generate_tuning_report(
    results: list[dict],
    config,
) -> str:
    """Generate tuning summary as a markdown report.

    Contains only tuning metadata — no evaluation metrics,
    no model comparison, no conclusions.

    Args:
        results: List of tuning result dictionaries.
        config: TuningConfig used for the search.

    Returns:
        Complete markdown report as a string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = config.to_dict()

    md = "# Hyperparameter Tuning Report\n\n"
    md += f"**Generated**: {timestamp}\n\n"
    md += "---\n\n"

    # GridSearchCV Configuration
    md += "## GridSearchCV Configuration\n\n"
    md += "| Parameter | Value |\n"
    md += "|-----------|-------|\n"
    for key, value in cfg.items():
        md += f"| {key} | `{value}` |\n"
    md += "\n---\n\n"

    # Summary Table
    md += "## Tuning Summary\n\n"
    md += "| Model | Best Score | Candidates | CV Iterations | Duration (s) | Status |\n"
    md += "|-------|-----------|------------|---------------|-------------|--------|\n"
    for r in results:
        md += (
            f"| {r['model_name']} | {r['best_score']:.4f} | "
            f"{r['n_candidates']} | {r['n_cv_iterations']} | "
            f"{r['duration']:.2f} | ✓ Completed |\n"
        )
    md += "\n---\n\n"

    # Per-model details
    for i, r in enumerate(results, 1):
        md += f"## {i}. {r['model_name']}\n\n"

        # Parameter Grid
        md += "### Parameter Grid\n\n"
        md += "| Parameter | Search Values |\n"
        md += "|-----------|---------------|\n"
        for param, values in r["param_grid"].items():
            md += f"| `{param}` | {values} |\n"
        md += "\n"

        # Best Parameters
        md += "### Best Parameters\n\n"
        md += "| Parameter | Best Value |\n"
        md += "|-----------|------------|\n"
        for param, value in r["best_params"].items():
            md += f"| `{param}` | `{value}` |\n"
        md += "\n"

        # Best Score
        md += f"**Best CV Score ({cfg['scoring']})**: `{r['best_score']:.6f}`\n\n"
        md += f"**Search Duration**: {r['duration']:.2f}s\n\n"
        md += f"**Candidates Tested**: {r['n_candidates']}\n\n"
        md += f"**Total CV Fits**: {r['n_cv_iterations']}\n\n"
        md += "---\n\n"

    # Notes
    md += "## Notes\n\n"
    md += "- All scores are **cross-validation scores on the training set**.\n"
    md += "- The test set was NOT used during tuning.\n"
    md += "- No evaluation metrics are included in this report.\n"
    md += "- Model evaluation will be performed in `06_model_evaluation.ipynb`.\n"
    md += "\n---\n\n"
    md += "*Report generated automatically by the tuning pipeline.*\n"

    return md


def export_tuning_statistics(
    results: list[dict],
    filepath: str,
) -> None:
    """Export tuning statistics as CSV.

    Args:
        results: List of tuning result dictionaries.
        filepath: Destination CSV path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            "model_name": r["model_name"],
            "best_score": r["best_score"],
            "best_params": str(r["best_params"]),
            "n_candidates": r["n_candidates"],
            "n_cv_iterations": r["n_cv_iterations"],
            "duration_seconds": r["duration"],
        })

    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"  ✓ Tuning statistics exported: {filepath}")


def save_tuning_report(content: str, filepath: str) -> None:
    """Save tuning report to file.

    Args:
        content: Markdown report string.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ Tuning report saved: {filepath}")
