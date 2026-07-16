"""Class-level error analysis.

Analyzes which classes suffer most from recall failures (false negatives)
and precision failures (false positives).
"""

import pandas as pd


def analyze_class_errors(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Analyze false positives and false negatives per class.

    Args:
        df: Main error analysis DataFrame.
        model_col: Name of the column containing model predictions.

    Returns:
        DataFrame with class-level error statistics.
    """
    classes = sorted(df["true_label"].unique())
    stats = []
    
    for c in classes:
        # True instances of this class
        actual_c = df[df["true_label"] == c]
        # Predicted instances of this class
        predicted_c = df[df[model_col] == c]
        
        # Recall failures (False Negatives): True class is c, but predicted otherwise
        fn = len(actual_c[actual_c[model_col] != c])
        
        # Precision failures (False Positives): Predicted c, but true class is otherwise
        fp = len(predicted_c[predicted_c["true_label"] != c])
        
        # Total actual
        support = len(actual_c)
        
        # Recall Error Rate (percentage of actual class missed)
        fn_rate = (fn / support) if support > 0 else 0.0
        
        stats.append({
            "class": c,
            "support": support,
            "false_negatives": fn,
            "false_positives": fp,
            "recall_error_rate": fn_rate,
        })
        
    res_df = pd.DataFrame(stats)
    res_df.set_index("class", inplace=True)
    res_df.sort_values("recall_error_rate", ascending=False, inplace=True)
    
    return res_df


def identify_difficult_classes(class_errors_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate average recall error rate across all models to find hardest classes.

    Args:
        class_errors_dfs: Dictionary mapping model names to their class error DataFrames.

    Returns:
        Aggregated DataFrame ranked by difficulty.
    """
    combined = pd.DataFrame()
    
    for model_name, c_df in class_errors_dfs.items():
        combined[f"{model_name}_fn_rate"] = c_df["recall_error_rate"]
        
    combined["avg_recall_error_rate"] = combined.mean(axis=1)
    combined.sort_values("avg_recall_error_rate", ascending=False, inplace=True)
    
    return combined
