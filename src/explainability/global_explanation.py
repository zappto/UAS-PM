"""Global explanation module.

Provides functionalities to calculate global feature importance
across the entire dataset.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd


class GlobalExplanation:
    """Extracts global feature importance from SHAP values."""

    def __init__(self, feature_names: np.ndarray, class_names: List[str]):
        """Initialize with feature and class names.

        Args:
            feature_names: Array of feature names from vectorizer.
            class_names: List of class labels.
        """
        self.feature_names = feature_names
        self.class_names = class_names

    def get_global_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Calculate mean absolute SHAP values for global importance.

        Args:
            shap_values: Array of SHAP values.
                         Shape: (num_samples, num_features, num_classes) or (num_samples, num_features)

        Returns:
            pd.DataFrame: Global feature importance dataframe.
        """
        # Calculate mean absolute SHAP values across samples
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        df_data = {"feature": self.feature_names}
        
        if mean_abs_shap.ndim == 2:
            # Multiclass
            for i, class_name in enumerate(self.class_names):
                df_data[f"importance_{class_name}"] = mean_abs_shap[:, i]
            
            # Overall importance (sum across classes)
            df_data["importance_overall"] = mean_abs_shap.sum(axis=1)
        else:
            # Binary
            df_data["importance_overall"] = mean_abs_shap

        df = pd.DataFrame(df_data)
        df = df.sort_values(by="importance_overall", ascending=False).reset_index(drop=True)
        return df

    def get_top_features_per_class(self, global_importance_df: pd.DataFrame, top_k: int = 10) -> Dict[str, List[str]]:
        """Get top k features for each class based on global importance.

        Args:
            global_importance_df: DataFrame from get_global_importance.
            top_k: Number of features to return per class.

        Returns:
            Dict mapping class names to lists of top features.
        """
        top_features = {}
        for class_name in self.class_names:
            col_name = f"importance_{class_name}"
            if col_name in global_importance_df.columns:
                top = global_importance_df.nlargest(top_k, col_name)
                top_features[class_name] = top["feature"].tolist()
        
        return top_features
