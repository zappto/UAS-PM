"""Dataset validator for preprocessing stage.

Validates the dataset before preprocessing to catch quality issues
early. Checks for missing text, empty text, invalid labels,
and duplicate rows.
"""

import pandas as pd

from src.config.settings import VALID_LABELS


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate dataset before preprocessing.

    Performs comprehensive quality checks on the input dataset
    to ensure it meets the requirements for preprocessing.

    Args:
        df: Input DataFrame with 'text' and 'label' columns.

    Returns:
        Dictionary containing:
            - is_valid (bool): Whether the dataset passes all checks.
            - total_rows (int): Total number of rows.
            - issues (dict): Detailed issue counts.
            - invalid_label_values (list): Unique invalid labels found.
            - summary (str): Human-readable summary.

    Raises:
        ValueError: If required columns are missing.
    """
    required = ["text", "label"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    total_rows = len(df)

    # Missing text (null values)
    missing_text = int(df["text"].isnull().sum())

    # Empty text (whitespace only or empty string)
    text_as_str = df["text"].fillna("").astype(str)
    empty_text = int((text_as_str.str.strip() == "").sum())

    # Missing labels
    missing_label = int(df["label"].isnull().sum())

    # Invalid labels
    label_as_str = df["label"].fillna("").astype(str)
    invalid_mask = ~label_as_str.isin(VALID_LABELS)
    invalid_labels = int(invalid_mask.sum())
    invalid_label_values = sorted(
        label_as_str[invalid_mask].unique().tolist()
    )

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Duplicate text (same text, possibly different labels)
    duplicate_text = int(df["text"].duplicated().sum())

    issues = {
        "missing_text": missing_text,
        "empty_text": empty_text,
        "missing_label": missing_label,
        "invalid_labels": invalid_labels,
        "duplicate_rows": duplicate_rows,
        "duplicate_text": duplicate_text,
    }

    is_valid = all(v == 0 for v in issues.values())

    # Build summary
    lines = [f"Dataset Validation Report ({total_rows:,} rows)"]
    lines.append("-" * 50)
    for issue_name, count in issues.items():
        status = "✓" if count == 0 else f"⚠ {count:,}"
        lines.append(f"  {issue_name:>20}: {status}")
    lines.append("-" * 50)
    lines.append(f"  {'Status':>20}: {'PASS' if is_valid else 'HAS ISSUES'}")

    if invalid_label_values:
        lines.append(f"  Invalid labels found: {invalid_label_values}")

    summary = "\n".join(lines)

    return {
        "is_valid": is_valid,
        "total_rows": total_rows,
        "issues": issues,
        "invalid_label_values": invalid_label_values,
        "summary": summary,
    }
