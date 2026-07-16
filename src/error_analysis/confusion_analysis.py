"""Confusion pattern analysis.

Identifies which pairs of classes are most frequently confused
by the models.
"""

import pandas as pd


def analyze_confusions(df: pd.DataFrame, model_col: str, top_k: int = 10) -> pd.DataFrame:
    """Find the most frequently confused pairs of classes.

    Args:
        df: DataFrame containing 'true_label' and the model's prediction column.
        model_col: Name of the column containing model predictions.
        top_k: Number of top pairs to return.

    Returns:
        DataFrame of top confused pairs.
    """
    # Filter only incorrect predictions
    errors = df[df["true_label"] != df[model_col]]
    
    if len(errors) == 0:
        return pd.DataFrame()
        
    # Group by true label and predicted label to count occurrences
    confusion_counts = errors.groupby(["true_label", model_col]).size().reset_index(name="count")
    confusion_counts.rename(columns={model_col: "predicted_label"}, inplace=True)
    
    # Sort by count descending
    confusion_counts.sort_values("count", ascending=False, inplace=True)
    
    return confusion_counts.head(top_k)


def aggregate_top_confusions(confusion_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate confusion counts across all models to find universal confusions.

    Args:
        confusion_dfs: Dictionary mapping model names to their confusion DataFrames.

    Returns:
        Aggregated DataFrame of confusions.
    """
    all_confusions = []
    
    for model_name, conf_df in confusion_dfs.items():
        if not conf_df.empty:
            # Add model name column for tracking
            conf = conf_df.copy()
            conf["model"] = model_name
            all_confusions.append(conf)
            
    if not all_confusions:
        return pd.DataFrame()
        
    combined = pd.concat(all_confusions)
    
    # Sum counts across all models for the same pair
    aggregated = combined.groupby(["true_label", "predicted_label"])["count"].sum().reset_index()
    aggregated.sort_values("count", ascending=False, inplace=True)
    
    return aggregated
