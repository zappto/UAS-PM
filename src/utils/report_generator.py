"""EDA report generator — produces markdown summary and CSV exports."""

import os
from datetime import datetime

import pandas as pd

from src.config.settings import VALID_LABELS


def assess_data_quality(df: pd.DataFrame) -> dict[str, int]:
    """Assess data quality issues in the dataset.

    Checks for missing values, empty text, very short text,
    extremely long text, and invalid labels.

    Args:
        df: Input DataFrame with 'text' and 'label' columns.

    Returns:
        Dictionary with counts for each quality issue category.
    """
    text_col = df["text"].astype(str)
    word_counts = text_col.apply(lambda x: len(x.split()))

    return {
        "missing_text": int(df["text"].isnull().sum()),
        "missing_label": int(df["label"].isnull().sum()),
        "empty_text": int((text_col.str.strip() == "").sum()),
        "very_short_text": int((word_counts < 3).sum()),
        "extremely_long_text": int((word_counts > 500).sum()),
        "invalid_labels": int(
            (~df["label"].isin(VALID_LABELS)).sum()
        ),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def generate_eda_report(
    overview: dict,
    label_dist: pd.DataFrame,
    text_stats: dict,
    vocab_stats: dict,
    imbalance: dict,
    duplicate_info: dict,
    missing_df: pd.DataFrame,
    quality_issues: dict,
) -> str:
    """Generate a complete EDA summary as a markdown string.

    Args:
        overview: Dataset overview dictionary from get_dataset_overview().
        label_dist: Label distribution DataFrame.
        text_stats: Text statistics dictionary.
        vocab_stats: Vocabulary statistics dictionary.
        imbalance: Class imbalance assessment dictionary.
        duplicate_info: Duplicate count dictionary.
        missing_df: Missing values DataFrame.
        quality_issues: Data quality issues dictionary.

    Returns:
        Complete markdown report as a string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"# Exploratory Data Analysis (EDA) Report\n\n"
    md += f"**Generated**: {timestamp}\n\n"
    md += f"**Dataset**: `dataset/processed/final_dataset.csv`\n\n"
    md += "---\n\n"

    # ── Dataset Overview ─────────────────────────────────────────────────
    md += "## 1. Dataset Overview\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    md += f"| Total Rows | {overview['shape']['rows']:,} |\n"
    md += f"| Total Columns | {overview['shape']['columns']} |\n"
    for col, dtype in overview["data_types"].items():
        md += f"| Data Type (`{col}`) | {dtype} |\n"
    md += f"| Memory Usage | {overview['memory_usage']['total_mb']} MB |\n"
    md += "\n---\n\n"

    # ── Data Quality ─────────────────────────────────────────────────────
    md += "## 2. Data Quality\n\n"
    md += "### Missing Values\n\n"
    md += "| Column | Missing Count | Missing Percentage |\n"
    md += "|--------|---------------|--------------------|\n"
    for _, row in missing_df.iterrows():
        md += f"| {row['column']} | {int(row['missing_count'])} | {row['missing_percentage']}% |\n"

    md += "\n### Duplicate Analysis\n\n"
    md += f"| Metric | Value |\n"
    md += f"|--------|-------|\n"
    md += f"| Duplicate Rows | {duplicate_info['duplicate_count']:,} |\n"
    md += f"| Duplicate Percentage | {duplicate_info['duplicate_percentage']}% |\n"
    md += f"| Total Rows | {duplicate_info['total_rows']:,} |\n"

    md += "\n### Data Quality Issues\n\n"
    md += "| Issue | Count |\n"
    md += "|-------|-------|\n"
    issue_labels = {
        "missing_text": "Missing Text",
        "missing_label": "Missing Label",
        "empty_text": "Empty Text (whitespace only)",
        "very_short_text": "Very Short Text (< 3 words)",
        "extremely_long_text": "Extremely Long Text (> 500 words)",
        "invalid_labels": "Invalid Labels",
        "duplicate_rows": "Duplicate Rows",
    }
    for key, label in issue_labels.items():
        count = quality_issues.get(key, 0)
        status = "✓" if count == 0 else f"⚠ {count:,}"
        md += f"| {label} | {status} |\n"
    md += "\n---\n\n"

    # ── Label Distribution ───────────────────────────────────────────────
    md += "## 3. Label Distribution\n\n"
    md += "| Label | Count | Percentage |\n"
    md += "|-------|-------|------------|\n"
    for _, row in label_dist.iterrows():
        md += f"| `{row['label']}` | {int(row['count']):,} | {row['percentage']}% |\n"

    md += "\n### Class Imbalance Assessment\n\n"
    md += f"| Metric | Value |\n"
    md += f"|--------|-------|\n"
    md += f"| Imbalance Ratio | {imbalance['ratio']}:1 |\n"
    md += f"| Status | **{imbalance['status']}** |\n"
    md += f"| Largest Class | `{imbalance['max_class']}` ({imbalance['max_count']:,}) |\n"
    md += f"| Smallest Class | `{imbalance['min_class']}` ({imbalance['min_count']:,}) |\n"
    md += f"\n> {imbalance['explanation']}\n"
    md += "\n---\n\n"

    # ── Text Statistics ──────────────────────────────────────────────────
    md += "## 4. Text Statistics\n\n"
    md += "### Word Count\n\n"
    md += "| Statistic | Value |\n"
    md += "|-----------|-------|\n"
    wc = text_stats.get("word_count", {})
    for stat_name in ["mean", "median", "mode", "std", "min", "max"]:
        md += f"| {stat_name.capitalize()} | {wc.get(stat_name, 'N/A')} |\n"

    md += "\n### Character Count\n\n"
    md += "| Statistic | Value |\n"
    md += "|-----------|-------|\n"
    cc = text_stats.get("char_count", {})
    for stat_name in ["mean", "median", "mode", "std", "min", "max"]:
        md += f"| {stat_name.capitalize()} | {cc.get(stat_name, 'N/A')} |\n"

    md += "\n### Vocabulary\n\n"
    md += "| Metric | Value |\n"
    md += "|--------|-------|\n"
    md += f"| Unique Words (Vocabulary Size) | {vocab_stats['unique_words']:,} |\n"
    md += f"| Total Words | {vocab_stats['total_words']:,} |\n"
    md += "\n---\n\n"

    # ── Research Findings ────────────────────────────────────────────────
    md += "## 5. Research Findings\n\n"

    findings = _generate_findings(
        overview, label_dist, text_stats, vocab_stats,
        imbalance, duplicate_info, quality_issues,
    )
    for finding in findings:
        md += f"- {finding}\n"
    md += "\n---\n\n"

    # ── Recommendations ──────────────────────────────────────────────────
    md += "## 6. Recommendations before Preprocessing\n\n"
    recs = _generate_recommendations(
        imbalance, quality_issues, text_stats, vocab_stats,
    )
    for i, rec in enumerate(recs, 1):
        md += f"{i}. {rec}\n"
    md += "\n---\n\n"
    md += f"*Report generated automatically by the EDA pipeline.*\n"

    return md


def _generate_findings(
    overview: dict,
    label_dist: pd.DataFrame,
    text_stats: dict,
    vocab_stats: dict,
    imbalance: dict,
    duplicate_info: dict,
    quality_issues: dict,
) -> list[str]:
    """Generate research findings as bullet points.

    Args:
        All analysis result dictionaries.

    Returns:
        List of finding strings.
    """
    findings = []

    # Dataset size
    total = overview["shape"]["rows"]
    findings.append(f"Dataset contains {total:,} samples across "
                    f"{len(label_dist)} classes.")

    # Imbalance
    findings.append(f"Class imbalance status: **{imbalance['status']}** "
                    f"(ratio {imbalance['ratio']}:1).")

    # Dominant class
    top_label = label_dist.iloc[0]
    findings.append(
        f"The dominant class is `{top_label['label']}` with "
        f"{int(top_label['count']):,} samples ({top_label['percentage']}%)."
    )

    # Smallest class
    bottom_label = label_dist.iloc[-1]
    findings.append(
        f"The smallest class is `{bottom_label['label']}` with "
        f"{int(bottom_label['count']):,} samples ({bottom_label['percentage']}%)."
    )

    # Text length
    wc = text_stats.get("word_count", {})
    findings.append(
        f"Average text length is {wc.get('mean', 'N/A')} words "
        f"(median: {wc.get('median', 'N/A')}, std: {wc.get('std', 'N/A')})."
    )

    # Vocabulary
    findings.append(
        f"Vocabulary size is {vocab_stats['unique_words']:,} unique words "
        f"from {vocab_stats['total_words']:,} total words."
    )

    # Duplicates
    if duplicate_info["duplicate_count"] > 0:
        findings.append(
            f"Dataset contains {duplicate_info['duplicate_count']:,} "
            f"duplicate rows ({duplicate_info['duplicate_percentage']}%)."
        )
    else:
        findings.append("Dataset contains no duplicate rows.")

    # Quality issues
    short = quality_issues.get("very_short_text", 0)
    if short > 0:
        findings.append(
            f"There are {short:,} very short texts (< 3 words) that may "
            f"need attention during preprocessing."
        )

    long_count = quality_issues.get("extremely_long_text", 0)
    if long_count > 0:
        findings.append(
            f"There are {long_count:,} extremely long texts (> 500 words) "
            f"that may affect TF-IDF performance."
        )

    return findings


def _generate_recommendations(
    imbalance: dict,
    quality_issues: dict,
    text_stats: dict,
    vocab_stats: dict,
) -> list[str]:
    """Generate preprocessing recommendations.

    Args:
        All analysis result dictionaries.

    Returns:
        List of recommendation strings.
    """
    recs = []

    # Imbalance
    if imbalance["status"] == "Severely Imbalanced":
        recs.append(
            "**Handle class imbalance**: Consider applying class weights, "
            "oversampling (SMOTE), or undersampling to address the severe "
            f"imbalance (ratio {imbalance['ratio']}:1)."
        )
    elif imbalance["status"] == "Mildly Imbalanced":
        recs.append(
            "**Monitor class imbalance**: Use stratified train/test split "
            "and evaluate using F1-Score (macro/weighted) rather than accuracy "
            "alone."
        )

    # Preprocessing steps
    recs.append(
        "**Text cleaning**: Remove URLs, mentions, hashtags, emojis, HTML "
        "tags, punctuation, numbers, and extra whitespace as specified in the DRD."
    )
    recs.append(
        "**Text preprocessing**: Apply case folding → tokenization → "
        "stopword removal → stemming (Sastrawi) → join tokens."
    )

    # Short texts
    short = quality_issues.get("very_short_text", 0)
    if short > 0:
        recs.append(
            f"**Review short texts**: {short:,} texts have fewer than 3 words. "
            f"Verify they contain meaningful content before TF-IDF extraction."
        )

    # TF-IDF
    recs.append(
        f"**TF-IDF tuning**: With a vocabulary of {vocab_stats['unique_words']:,} "
        f"unique words, consider setting max_features and min_df/max_df "
        f"parameters to reduce dimensionality."
    )

    # Split
    recs.append(
        "**Data split**: Use 80/20 stratified split with random_state=42 "
        "as defined in the TRD."
    )

    return recs


def save_report(content: str, filepath: str) -> None:
    """Save markdown report to file.

    Args:
        content: Markdown string to save.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ Report saved: {filepath}")


def export_dataset_overview(overview: dict, filepath: str) -> None:
    """Export dataset overview as CSV.

    Args:
        overview: Overview dictionary from get_dataset_overview().
        filepath: Destination CSV path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    rows = [
        {"metric": "total_rows", "value": overview["shape"]["rows"]},
        {"metric": "total_columns", "value": overview["shape"]["columns"]},
        {"metric": "memory_mb", "value": overview["memory_usage"]["total_mb"]},
        {"metric": "duplicate_count", "value": overview["duplicates"]["duplicate_count"]},
        {"metric": "duplicate_percentage", "value": overview["duplicates"]["duplicate_percentage"]},
    ]
    for col, dtype in overview["data_types"].items():
        rows.append({"metric": f"dtype_{col}", "value": dtype})
    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"  ✓ Exported: {filepath}")


def export_label_distribution(dist_df: pd.DataFrame, filepath: str) -> None:
    """Export label distribution as CSV.

    Args:
        dist_df: Label distribution DataFrame.
        filepath: Destination CSV path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    dist_df.to_csv(filepath, index=False)
    print(f"  ✓ Exported: {filepath}")


def export_text_statistics(stats: dict, filepath: str) -> None:
    """Export text statistics as CSV.

    Args:
        stats: Text statistics dictionary.
        filepath: Destination CSV path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    rows = []
    for metric_group, values in stats.items():
        for stat_name, stat_val in values.items():
            rows.append({
                "metric": f"{metric_group}_{stat_name}",
                "value": stat_val,
            })
    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"  ✓ Exported: {filepath}")


def export_vocabulary_statistics(vocab: dict, filepath: str) -> None:
    """Export vocabulary statistics as CSV.

    Args:
        vocab: Vocabulary statistics dictionary.
        filepath: Destination CSV path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    rows = [{"metric": k, "value": v} for k, v in vocab.items()]
    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"  ✓ Exported: {filepath}")
