"""ROC Curve and ROC-AUC calculation for multiclass classification.

Uses a One-vs-Rest (OvR) approach to calculate ROC for each class
independently and plots them together using matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def calculate_roc_auc(
    y_true,
    y_prob: np.ndarray,
    labels: list[str],
) -> dict:
    """Calculate ROC curves and AUC for each class (OvR).

    Args:
        y_true: True labels (strings).
        y_prob: Predicted probabilities of shape (n_samples, n_classes).
        labels: List of class labels in the same order as model.classes_.

    Returns:
        Dictionary containing fpr, tpr, and auc for each class,
        plus the macro-average.
    """
    # Binarize labels for One-vs-Rest calculations
    y_true_bin = label_binarize(y_true, classes=labels)
    n_classes = len(labels)

    results = {}
    
    # Calculate for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        results[labels[i]] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc
        }

    # Calculate macro-average ROC curve
    all_fpr = np.unique(np.concatenate([results[label]["fpr"] for label in labels]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for label in labels:
        mean_tpr += np.interp(all_fpr, results[label]["fpr"], results[label]["tpr"])
        
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    results["macro"] = {
        "fpr": all_fpr,
        "tpr": mean_tpr,
        "auc": macro_auc
    }

    return results


def plot_roc_curve(
    roc_data: dict,
    labels: list[str],
    model_name: str,
    output_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot multiclass ROC curves (OvR).

    Args:
        roc_data: Output from calculate_roc_auc.
        labels: List of class labels.
        model_name: Name of the model.
        output_path: Path to save the figure (optional).
        dpi: Resolution for saved figure.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot macro-average ROC curve first (thicker line)
    ax.plot(
        roc_data["macro"]["fpr"],
        roc_data["macro"]["tpr"],
        label=f"Macro-average ROC (AUC = {roc_data['macro']['auc']:.3f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    # Plot ROC curve for each class
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for label, color in zip(labels, colors):
        ax.plot(
            roc_data[label]["fpr"],
            roc_data[label]["tpr"],
            color=color,
            lw=2,
            label=f"ROC of {label} (AUC = {roc_data[label]['auc']:.3f})",
        )

    # Plot random chance line
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Chance")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"Receiver Operating Characteristic (OvR): {model_name}", fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  ✓ ROC curve plot saved: {output_path}")

    plt.close(fig)
    return fig
