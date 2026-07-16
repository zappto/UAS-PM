"""Visualization functions for EDA — all return matplotlib Figure objects."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure

# ─── Global style ────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

COLORS = sns.color_palette("Set2")


def plot_label_distribution(distribution_df: pd.DataFrame) -> Figure:
    """Create a horizontal bar chart of label distribution.

    Args:
        distribution_df: DataFrame with 'label' and 'count' columns.

    Returns:
        Matplotlib Figure object.
    """
    df = distribution_df.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df["label"], df["count"], color=COLORS[:len(df)])
    for bar, val in zip(bars, df["count"]):
        ax.text(
            bar.get_width() + max(df["count"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Count")
    ax.set_ylabel("Label")
    ax.set_title("Label Distribution")
    fig.tight_layout()
    return fig


def plot_label_distribution_pie(distribution_df: pd.DataFrame) -> Figure:
    """Create a pie chart of label distribution.

    Args:
        distribution_df: DataFrame with 'label' and 'count' columns.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = COLORS[:len(distribution_df)]
    wedges, texts, autotexts = ax.pie(
        distribution_df["count"],
        labels=distribution_df["label"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.85,
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
    ax.set_title("Label Distribution (Percentage)")
    fig.tight_layout()
    return fig


def plot_text_length_histogram(
    df: pd.DataFrame, col: str = "word_count"
) -> Figure:
    """Create a histogram of text lengths with mean/median lines.

    Args:
        df: DataFrame containing the length column.
        col: Column name for text length values.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    data = df[col]
    ax.hist(data, bins=50, color=COLORS[0], edgecolor="white", alpha=0.8)
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color=COLORS[3], linestyle="--", linewidth=2,
               label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color=COLORS[1], linestyle="--", linewidth=2,
               label=f"Median: {median_val:.1f}")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Text Length Distribution (Word Count)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_text_length_boxplot(
    df: pd.DataFrame, col: str = "word_count"
) -> Figure:
    """Create a box plot of text length per label.

    Args:
        df: DataFrame with the length column and 'label' column.
        col: Column name for text length values.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby("label")[col].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="label", y=col, hue="label", order=order,
                palette="Set2", ax=ax, legend=False)
    ax.set_xlabel("Label")
    ax.set_ylabel("Word Count")
    ax.set_title("Text Length Distribution by Label")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_top_words(
    word_counts: list[tuple[str, int]],
    title: str = "Top 20 Most Frequent Words",
) -> Figure:
    """Create a horizontal bar chart of top words.

    Args:
        word_counts: List of (word, count) tuples.
        title: Chart title.

    Returns:
        Matplotlib Figure object.
    """
    words = [w for w, _ in reversed(word_counts)]
    counts = [c for _, c in reversed(word_counts)]
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(words, counts, color=COLORS[0])
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_top_ngrams(
    ngram_counts: list[tuple[str, int]],
    title: str = "Top 20 Bigrams",
) -> Figure:
    """Create a horizontal bar chart of top n-grams.

    Args:
        ngram_counts: List of (ngram_string, count) tuples.
        title: Chart title.

    Returns:
        Matplotlib Figure object.
    """
    ngrams = [ng for ng, _ in reversed(ngram_counts)]
    counts = [c for _, c in reversed(ngram_counts)]
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(ngrams, counts, color=COLORS[2])
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_missing_values(missing_df: pd.DataFrame) -> Figure:
    """Create a bar chart of missing values per column.

    Args:
        missing_df: DataFrame with 'column' and 'missing_count' columns.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        missing_df["column"],
        missing_df["missing_count"],
        color=COLORS[3],
        edgecolor="white",
    )
    for bar, val in zip(bars, missing_df["missing_count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            fontsize=11,
        )
    if missing_df["missing_count"].sum() == 0:
        ax.text(
            0.5, 0.5, "No missing values found ✓",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color="green", fontweight="bold",
        )
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")
    ax.set_title("Missing Values per Column")
    fig.tight_layout()
    return fig


def plot_duplicate_summary(duplicate_info: dict) -> Figure:
    """Create a bar chart showing unique vs duplicate rows.

    Args:
        duplicate_info: Dict with 'duplicate_count' and 'total_rows' keys.

    Returns:
        Matplotlib Figure object.
    """
    unique = duplicate_info["total_rows"] - duplicate_info["duplicate_count"]
    dup = duplicate_info["duplicate_count"]
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Unique", "Duplicate"]
    values = [unique, dup]
    colors_bar = ["#66c2a5", "#fc8d62"]
    bars = ax.bar(categories, values, color=colors_bar, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:,}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("Count")
    ax.set_title("Duplicate Analysis")
    fig.tight_layout()
    return fig
