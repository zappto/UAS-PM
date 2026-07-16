"""Confusion matrix generation and visualization.

Generates standard confusion matrices and plots custom heatmaps
using only matplotlib (no seaborn), as per project requirements.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def generate_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate confusion matrix array and DataFrame.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of class labels.

    Returns:
        Tuple of (numpy array, pandas DataFrame).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create DataFrame for easier CSV export
    class_labels = labels if labels is not None else sorted(list(set(y_true)))
    df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    
    return cm, df


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    model_name: str,
    output_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot confusion matrix heatmap using pure matplotlib.

    Replicates the look of seaborn heatmaps without the dependency.

    Args:
        cm: Confusion matrix numpy array.
        labels: List of class labels.
        model_name: Name of the model for the title.
        output_path: Path to save the figure (optional).
        dpi: Resolution for saved figure.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set labels and title
    ax.set_title(f"Confusion Matrix: {model_name}", pad=20, fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)

    # Setup ticks
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    
    # Rotate x labels for better readability
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Add text annotations inside cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  ✓ Confusion matrix plot saved: {output_path}")

    plt.close(fig)
    return fig
