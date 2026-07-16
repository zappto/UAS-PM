"""Label distribution and class imbalance analysis."""

import pandas as pd


def get_label_distribution(
    df: pd.DataFrame, col: str = "label"
) -> pd.DataFrame:
    """Compute label distribution with counts and percentages.

    Args:
        df: Input DataFrame.
        col: Name of the label column.

    Returns:
        DataFrame with columns: label, count, percentage.
        Sorted by count descending.
    """
    counts = df[col].value_counts()
    total = len(df)
    dist = pd.DataFrame({
        "label": counts.index,
        "count": counts.values,
        "percentage": (counts.values / total * 100).round(2),
    })
    return dist.reset_index(drop=True)


def get_class_count(df: pd.DataFrame, col: str = "label") -> int:
    """Get the number of unique classes.

    Args:
        df: Input DataFrame.
        col: Name of the label column.

    Returns:
        Number of unique label values.
    """
    return df[col].nunique()


def get_class_imbalance_ratio(
    df: pd.DataFrame, col: str = "label"
) -> dict[str, float | str]:
    """Assess class imbalance in the dataset.

    Args:
        df: Input DataFrame.
        col: Name of the label column.

    Returns:
        Dictionary with:
            - ratio (float): Max class count / min class count.
            - status (str): 'Balanced', 'Mildly Imbalanced',
              or 'Severely Imbalanced'.
            - max_class (str): Label of the largest class.
            - min_class (str): Label of the smallest class.
            - max_count (int): Count of the largest class.
            - min_count (int): Count of the smallest class.
            - explanation (str): Human-readable explanation.
    """
    counts = df[col].value_counts()
    max_count = int(counts.max())
    min_count = int(counts.min())
    max_class = str(counts.idxmax())
    min_class = str(counts.idxmin())
    ratio = round(max_count / min_count, 2) if min_count > 0 else float("inf")

    if ratio <= 2:
        status = "Balanced"
        explanation = (
            f"The dataset is balanced. The ratio between the largest class "
            f"('{max_class}': {max_count:,}) and the smallest class "
            f"('{min_class}': {min_count:,}) is {ratio}:1, which is within "
            f"the acceptable range (≤ 2:1)."
        )
    elif ratio <= 10:
        status = "Mildly Imbalanced"
        explanation = (
            f"The dataset is mildly imbalanced. The ratio between the largest "
            f"class ('{max_class}': {max_count:,}) and the smallest class "
            f"('{min_class}': {min_count:,}) is {ratio}:1. Consider using "
            f"stratified split and weighted metrics (F1-Score) for evaluation."
        )
    else:
        status = "Severely Imbalanced"
        explanation = (
            f"The dataset is severely imbalanced. The ratio between the "
            f"largest class ('{max_class}': {max_count:,}) and the smallest "
            f"class ('{min_class}': {min_count:,}) is {ratio}:1. Consider "
            f"oversampling, undersampling, class weights, or SMOTE. Use "
            f"stratified split and prioritize F1-Score over accuracy."
        )

    return {
        "ratio": ratio,
        "status": status,
        "max_class": max_class,
        "min_class": min_class,
        "max_count": max_count,
        "min_count": min_count,
        "explanation": explanation,
    }
