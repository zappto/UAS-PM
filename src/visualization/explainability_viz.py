"""Explainability visualization module.

Generates Matplotlib plots for explaining model predictions.
Note: Seaborn is explicitly avoided as per requirements.
"""

import os
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ExplainabilityVisualizer:
    """Generates explainability plots using Matplotlib."""

    def __init__(self, config: Any = None):
        """Initialize the visualizer.

        Args:
            config: ExplainabilityConfig instance.
        """
        self.dpi = config.dpi if config else 300
        self.fmt = config.figure_format if config else "png"
        self.max_features = config.max_features if config else 20
        self.figures_dir = config.figures_directory if config else "reports/explainability/figures"

    def _save_fig(self, filename: str) -> None:
        """Helper to save the current matplotlib figure."""
        os.makedirs(self.figures_dir, exist_ok=True)
        filepath = os.path.join(self.figures_dir, f"{filename}.{self.fmt}")
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def plot_global_feature_importance(self, importance_df: pd.DataFrame, filename: str = "feature_importance") -> None:
        """Plot global feature importance.

        Args:
            importance_df: DataFrame containing 'feature' and 'importance_overall'.
            filename: Output filename without extension.
        """
        top_df = importance_df.head(self.max_features).sort_values("importance_overall", ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(top_df["feature"], top_df["importance_overall"], color="skyblue")
        plt.xlabel("Mean Absolute Contribution / Importance")
        plt.ylabel("Features")
        plt.title("Global Feature Importance")
        self._save_fig(filename)

    def plot_local_explanation(self, local_exp: Dict[str, Any], filename: str = "local_explanation") -> None:
        """Plot local explanation (positive and negative contributions).

        Args:
            local_exp: Dictionary containing 'positive_features' and 'negative_features'.
            filename: Output filename without extension.
        """
        pos = local_exp.get("positive_features", [])
        neg = local_exp.get("negative_features", [])
        
        # Combine and sort by absolute score
        all_features = pos + neg
        all_features = sorted(all_features, key=lambda x: abs(x["score"]), reverse=False)
        
        features = [x["feature"] for x in all_features]
        scores = [x["score"] for x in all_features]
        colors = ["green" if s > 0 else "red" for s in scores]

        plt.figure(figsize=(10, 6))
        plt.barh(features, scores, color=colors)
        plt.xlabel("Contribution to Prediction")
        plt.ylabel("Features")
        plt.title(f"Local Explanation (Predicted Class: {local_exp.get('predicted_class_index', 'N/A')})")
        plt.axvline(x=0, color='black', linewidth=1)
        self._save_fig(filename)

    def plot_class_importance(self, class_features: Dict[str, List[str]], filename: str = "class_importance") -> None:
        """Plot important features per class.
        
        Generates a summary of top words for each class.

        Args:
            class_features: Dictionary mapping class names to lists of top features.
            filename: Output filename without extension.
        """
        classes = list(class_features.keys())
        n_classes = len(classes)
        
        fig, axes = plt.subplots(n_classes, 1, figsize=(8, 3 * n_classes))
        if n_classes == 1:
            axes = [axes]
            
        for ax, class_name in zip(axes, classes):
            features = class_features[class_name][::-1] # Reverse for plotting top-down
            # We just plot dummy bars to show the words since we only have the names here
            ax.barh(features, [1]*len(features), color="lightgray")
            ax.set_title(f"Top Features for Class: {class_name}")
            ax.set_xticks([])
            
        self._save_fig(filename)

    def plot_shap_summary(self, shap_values: np.ndarray, x_data: Any, feature_names: np.ndarray, filename: str = "shap_summary") -> None:
        """Plot SHAP summary plot using matplotlib.
        
        Args:
            shap_values: Array of SHAP values.
            x_data: Input data array.
            feature_names: Array of feature names.
            filename: Output filename.
        """
        import shap
        plt.figure()
        # Fallback to shap library plot, but ensure it saves properly
        # SHAP plots directly to matplotlib's current figure
        if shap_values.ndim == 3:
            # Multiclass
            shap.summary_plot(list(shap_values.transpose(2,0,1)), x_data, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, x_data, feature_names=feature_names, show=False)
        self._save_fig(filename)

    def plot_shap_bar(self, shap_values: np.ndarray, x_data: Any, feature_names: np.ndarray, filename: str = "shap_bar") -> None:
        """Plot SHAP bar plot using matplotlib.
        
        Args:
            shap_values: Array of SHAP values.
            x_data: Input data array.
            feature_names: Array of feature names.
            filename: Output filename.
        """
        import shap
        plt.figure()
        if shap_values.ndim == 3:
            shap.summary_plot(list(shap_values.transpose(2,0,1)), x_data, feature_names=feature_names, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, x_data, feature_names=feature_names, plot_type="bar", show=False)
        self._save_fig(filename)
