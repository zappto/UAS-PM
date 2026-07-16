"""Persistence module for explainability artifacts.

Handles saving data to various formats like CSV, JSON, NPY, and Joblib.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict


class Persistence:
    """Handles saving explanation results."""

    @staticmethod
    def save_csv(df: pd.DataFrame, filepath: str) -> None:
        """Save a pandas DataFrame to CSV.

        Args:
            df: The dataframe to save.
            filepath: Target file path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str) -> None:
        """Save a dictionary to JSON.

        Args:
            data: Dictionary to save.
            filepath: Target file path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def save_numpy(array: np.ndarray, filepath: str) -> None:
        """Save a NumPy array.

        Args:
            array: NumPy array to save.
            filepath: Target file path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, array)

    @staticmethod
    def save_joblib(obj: Any, filepath: str) -> None:
        """Save an object using joblib.

        Args:
            obj: Python object to save.
            filepath: Target file path.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(obj, filepath)

    @staticmethod
    def load_joblib(filepath: str) -> Any:
        """Load an object using joblib.

        Args:
            filepath: Path to the joblib file.

        Returns:
            The loaded object.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return joblib.load(filepath)
