"""Preprocessing report generator.

Generates markdown summary and CSV statistics for the
preprocessing stage, documenting all transformations applied.
"""

import os
from datetime import datetime

import pandas as pd

from src.config.preprocessing_config import PreprocessingConfig


def generate_preprocessing_report(
    total_rows: int,
    processed_rows: int,
    removed_empty: int,
    removed_duplicates: int,
    avg_length_before: float,
    avg_length_after: float,
    examples: list[dict[str, str]],
    processing_time: float,
    config: PreprocessingConfig,
    extra_notes: list[str] | None = None,
) -> str:
    """Generate the preprocessing summary as a markdown string.

    Args:
        total_rows: Total rows in the original dataset.
        processed_rows: Number of rows after processing.
        removed_empty: Number of rows removed due to empty text.
        removed_duplicates: Number of duplicate rows removed.
        avg_length_before: Average word count before preprocessing.
        avg_length_after: Average word count after preprocessing.
        examples: List of before/after transformation dicts.
        processing_time: Total processing time in seconds.
        config: The PreprocessingConfig used for this run.
        extra_notes: Optional additional notes to include.

    Returns:
        Complete markdown report as a string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_steps = config.get_active_steps()

    md = "# Preprocessing Summary Report\n\n"
    md += f"**Generated**: {timestamp}\n\n"
    md += "---\n\n"

    # Overview
    md += "## Overview\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    md += f"| Total Rows (Input) | {total_rows:,} |\n"
    md += f"| Processed Rows (Output) | {processed_rows:,} |\n"
    md += f"| Removed Empty Rows | {removed_empty:,} |\n"
    md += f"| Removed Duplicate Rows | {removed_duplicates:,} |\n"
    md += f"| Avg Text Length Before (words) | {avg_length_before} |\n"
    md += f"| Avg Text Length After (words) | {avg_length_after} |\n"
    md += f"| Processing Time | {processing_time:.2f} seconds |\n"

    md += "\n---\n\n"

    # Active steps
    md += "## Active Preprocessing Steps\n\n"
    for i, step in enumerate(active_steps, 1):
        md += f"{i}. {step}\n"

    md += "\n---\n\n"

    # Pipeline order
    md += "## Pipeline Order\n\n"
    md += "```\n"
    md += "Raw Text → Case Folding → Text Cleaning → Tokenization\n"
    md += "→ Stopword Removal → Stemming → Join Tokens\n"
    md += "```\n"

    md += "\n---\n\n"

    # Examples
    md += "## Example Transformations\n\n"
    for i, ex in enumerate(examples, 1):
        md += f"### Example {i}\n\n"
        md += f"**Before**:\n```\n{ex['before']}\n```\n\n"
        md += f"**After**:\n```\n{ex['after']}\n```\n\n"

    md += "---\n\n"

    # Notes
    md += "## Notes\n\n"
    default_notes = [
        "Preprocessing was applied to the `text` column only.",
        "The `label` column was not modified.",
        "Results are saved in `text_clean` column.",
        "This dataset is ready for TF-IDF feature extraction.",
    ]
    all_notes = default_notes + (extra_notes or [])
    for note in all_notes:
        md += f"- {note}\n"

    md += "\n---\n\n"
    md += "*Report generated automatically by the preprocessing pipeline.*\n"

    return md


def export_preprocessing_statistics(
    stats: dict,
    filepath: str,
) -> None:
    """Export preprocessing statistics as a CSV file.

    Args:
        stats: Statistics dictionary from preprocess_dataframe().
        filepath: Destination CSV file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rows = []
    for key, value in stats.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        rows.append({"metric": key, "value": value})

    pd.DataFrame(rows).to_csv(filepath, index=False)


def save_report(content: str, filepath: str) -> None:
    """Save markdown report to file.

    Args:
        content: Markdown string to save.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
