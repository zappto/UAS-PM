"""Compare evaluated models against each other.

Aggregates metrics from all evaluated models into a single DataFrame,
ranks them by primary metric (F1 Macro), and plots comparison charts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_comparison_dataframe(results: list[dict]) -> pd.DataFrame:
    """Aggregate evaluation results into a single comparison table.

    Args:
        results: List of result dictionaries from evaluate_model.

    Returns:
        DataFrame containing all overall metrics per model, sorted
        by f1_macro (descending).
    """
    rows = []
    for r in results:
        row = {"Model": r["model_name"]}
        row.update(r["overall_metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index("Model", inplace=True)
    
    # Sort by primary metric (F1 Macro) per requirements
    df.sort_values(by="f1_macro", ascending=False, inplace=True)
    
    return df


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metric_name: str,
    title: str,
    output_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot a bar chart comparing models on a specific metric.

    Args:
        comparison_df: Output from generate_comparison_dataframe.
        metric_name: Column name in the DataFrame (e.g., 'f1_macro').
        title: Title for the chart.
        output_path: Path to save the figure (optional).
        dpi: Resolution for saved figure.

    Returns:
        The matplotlib Figure object.
    """
    if metric_name not in comparison_df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in comparison data.")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get data sorted by metric descending
    sorted_df = comparison_df.sort_values(by=metric_name, ascending=True)
    models = sorted_df.index
    values = sorted_df[metric_name]

    # Create horizontal bar chart
    bars = ax.barh(models, values, color="steelblue", height=0.6)

    # Set labels and formatting
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_xlim([0.0, 1.05])
    ax.grid(axis="x", alpha=0.3)

    # Add data labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold"
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  ✓ Comparison plot saved: {output_path}")

    plt.close(fig)
    return fig


def generate_all_comparisons(
    comparison_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Convenience function to generate standard comparison plots.

    Args:
        comparison_df: Output from generate_comparison_dataframe.
        output_dir: Directory to save the plots.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    plots = [
        ("f1_macro", "F1 Score (Macro Average) Comparison"),
        ("accuracy", "Accuracy Comparison"),
        ("precision_macro", "Precision (Macro Average) Comparison"),
        ("recall_macro", "Recall (Macro Average) Comparison"),
    ]

    for metric, title in plots:
        plot_path = os.path.join(output_dir, f"comparison_{metric}.png")
        plot_metric_comparison(comparison_df, metric, title, plot_path)
