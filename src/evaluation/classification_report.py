"""Classification report generation for model evaluation.

Generates detailed classification reports in dictionary, CSV,
and Markdown formats.
"""

import pandas as pd
from sklearn.metrics import classification_report


def generate_classification_report(
    y_true,
    y_pred,
    labels: list[str] | None = None,
) -> dict:
    """Generate classification report dictionary.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of class labels to evaluate.

    Returns:
        Dictionary containing classification report data.
    """
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )


def report_to_dataframe(report_dict: dict) -> pd.DataFrame:
    """Convert classification report dictionary to DataFrame.

    Args:
        report_dict: Output from generate_classification_report.

    Returns:
        Formatted pandas DataFrame.
    """
    df = pd.DataFrame(report_dict).transpose()
    df.index.name = "class"
    return df


def generate_markdown_report(report_dict: dict) -> str:
    """Convert classification report dictionary to Markdown string.

    Args:
        report_dict: Output from generate_classification_report.

    Returns:
        Markdown formatted table string.
    """
    df = report_to_dataframe(report_dict)
    
    # Format floats for markdown display
    formatters = {
        "precision": "{:.4f}".format,
        "recall": "{:.4f}".format,
        "f1-score": "{:.4f}".format,
        "support": "{:.0f}".format,
    }
    
    # Some rows might not have all columns (like accuracy), handle gracefully
    for col in df.columns:
        if col in formatters:
            df[col] = df[col].apply(lambda x: formatters[col](x) if pd.notnull(x) else "")

    md = "| Class | Precision | Recall | F1-Score | Support |\n"
    md += "|-------|-----------|--------|----------|---------|\n"

    for idx, row in df.iterrows():
        # Skip accuracy as it has a different format, handle separately or put at bottom
        if idx == "accuracy":
            continue
            
        md += f"| {idx} | {row.get('precision', '')} | {row.get('recall', '')} | {row.get('f1-score', '')} | {row.get('support', '')} |\n"

    # Add accuracy if it exists
    if "accuracy" in df.index:
        acc = report_dict["accuracy"]
        md += f"| **accuracy** | | | **{acc:.4f}** | |\n"

    return md
