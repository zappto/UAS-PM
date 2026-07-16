"""Data loader and orchestrator for error analysis.

Handles loading original dataset text, true labels, and predictions.
If predictions do not exist, it dynamically loads models to generate them.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any

from src.config.settings import (
    CLEAN_DATASET_PATH,
    EVAL_PREDICTIONS_DIR,
    NB_BEST_MODEL_PATH,
    LR_BEST_MODEL_PATH,
    SVM_BEST_MODEL_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    VALID_LABELS
)
from src.utils.load_features import load_features
from src.error_analysis.validator import validate_dataset_alignment, validate_labels


def _generate_and_save_predictions(
    model_path: str,
    X_test: Any,
    save_path: str,
) -> pd.Series:
    """Load model, predict, and save to CSV."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model missing: {model_path}. Run tuning stage first.")
    
    print(f"    Generating predictions using {os.path.basename(model_path)}...")
    model = joblib.load(model_path)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_series = pd.Series(y_pred, name="prediction")
    
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_pred_series.to_csv(save_path, index=False)
    print(f"    Saved predictions to {save_path}")
    
    return y_pred_series


def load_and_prepare_errors() -> pd.DataFrame:
    """Load dataset, true labels, and predictions for all models.

    Matches predictions to the original clean text for analysis.
    If prediction CSVs are missing, dynamically generates them.

    Returns:
        DataFrame containing text, true_label, and prediction columns
        for each model.
    """
    print("Loading data for Error Analysis...")
    
    # 1. Load original clean text and test indices
    # We need to map X_test back to original text. 
    # load_features() loads processed sparse matrices.
    # But wait, how do we get the original text for X_test? 
    # Usually we rely on y_test's index to match the clean dataset.
    print("  Loading labels and text using exact split...")
    if not os.path.exists(CLEAN_DATASET_PATH):
        raise FileNotFoundError(f"Clean dataset missing: {CLEAN_DATASET_PATH}")
        
    clean_df = pd.read_csv(CLEAN_DATASET_PATH)
    
    # We must replicate the exact split to get the corresponding text
    from src.feature_engineering.train_test_split import split_dataset
    from src.config.tfidf_config import TfidfConfig
    
    _, test_text, _, y_test = split_dataset(
        df=clean_df,
        text_col="text_clean",
        label_col="label",
        config=TfidfConfig() # defaults to random_state=42
    )
    
    # Load the sparse matrix for prediction if needed
    _, X_test_sparse, _, _ = load_features()

    validate_labels(y_test, VALID_LABELS)

    # 2. Load or generate predictions
    models = {
        "naive_bayes": NB_BEST_MODEL_PATH,
        "logistic_regression": LR_BEST_MODEL_PATH,
        "svm": SVM_BEST_MODEL_PATH,
    }
    
    predictions = {}
    
    for name, model_path in models.items():
        pred_path = os.path.join(EVAL_PREDICTIONS_DIR, f"{name}_predictions.csv")
        
        if os.path.exists(pred_path):
            print(f"  Loading existing predictions for {name}...")
            pred_df = pd.read_csv(pred_path)
            # Handle standard "prediction" column name or fallback to first column
            if "prediction" in pred_df.columns:
                predictions[name] = pred_df["prediction"]
            else:
                predictions[name] = pred_df.iloc[:, 0]
        else:
            print(f"  Predictions for {name} not found. Generating...")
            predictions[name] = _generate_and_save_predictions(model_path, X_test_sparse, pred_path)

    # 3. Validate alignment
    validate_dataset_alignment(clean_df, y_test, predictions)
    
    # 4. Construct final DataFrame
    result_df = pd.DataFrame({
        "text": test_text,
        "true_label": y_test,
        "pred_naive_bayes": predictions["naive_bayes"],
        "pred_logistic_regression": predictions["logistic_regression"],
        "pred_svm": predictions["svm"],
    })
    
    print(f"✓ Error Analysis dataset ready. ({len(result_df)} samples)")
    return result_df
