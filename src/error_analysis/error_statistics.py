"""Error statistics calculation.

Aggregates high-level statistical summaries for the error analysis report.
"""

import pandas as pd


def generate_error_statistics(misclass_results: dict) -> pd.DataFrame:
    """Generate high-level error statistics for all models.

    Args:
        misclass_results: Dictionary output from analyze_misclassifications.

    Returns:
        DataFrame summarizing error counts and percentages.
    """
    stats = []
    
    for model, data in misclass_results.items():
        total = data["correct_count"] + data["error_count"]
        error_perc = (data["error_count"] / total * 100) if total > 0 else 0
        correct_perc = (data["correct_count"] / total * 100) if total > 0 else 0
        
        stats.append({
            "model": model,
            "total_samples": total,
            "correct_predictions": data["correct_count"],
            "errors": data["error_count"],
            "correct_percentage": correct_perc,
            "error_percentage": error_perc,
        })
        
    df = pd.DataFrame(stats)
    df.set_index("model", inplace=True)
    return df
