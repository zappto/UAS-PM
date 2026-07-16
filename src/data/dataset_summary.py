"""Dataset summary and overview statistics."""

import pandas as pd


def get_dataset_shape(df: pd.DataFrame) -> dict[str, int]:
    """Get the shape of the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with 'rows' and 'columns' counts.
    """
    return {"rows": df.shape[0], "columns": df.shape[1]}


def get_data_types(df: pd.DataFrame) -> pd.Series:
    """Get data types for each column.

    Args:
        df: Input DataFrame.

    Returns:
        Series mapping column names to their dtypes.
    """
    return df.dtypes


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing value statistics per column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: column, missing_count, missing_percentage.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    result = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_pct.values,
    })
    return result


def get_duplicate_count(df: pd.DataFrame) -> dict[str, int | float]:
    """Count duplicate rows in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with duplicate_count, duplicate_percentage, total_rows.
    """
    dup_count = df.duplicated().sum()
    total = len(df)
    return {
        "duplicate_count": int(dup_count),
        "duplicate_percentage": round(dup_count / total * 100, 2) if total > 0 else 0.0,
        "total_rows": total,
    }


def get_memory_usage(df: pd.DataFrame) -> dict[str, int | float | dict]:
    """Get memory usage of the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with total_bytes, total_mb, and per_column breakdown.
    """
    mem = df.memory_usage(deep=True)
    total_bytes = int(mem.sum())
    per_col = {col: int(mem[col]) for col in df.columns}
    return {
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "per_column": per_col,
    }


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Generate a comprehensive dataset overview.

    Combines shape, data types, missing values, duplicates,
    and memory usage into a single summary dictionary.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing all overview metrics.
    """
    shape = get_dataset_shape(df)
    dtypes = get_data_types(df)
    missing = get_missing_values(df)
    duplicates = get_duplicate_count(df)
    memory = get_memory_usage(df)

    return {
        "shape": shape,
        "data_types": {col: str(dtype) for col, dtype in dtypes.items()},
        "missing_values": missing,
        "duplicates": duplicates,
        "memory_usage": memory,
    }
