"""Project configuration and constants."""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATASET_PATH: str = os.path.join(BASE_DIR, "dataset", "processed", "final_dataset.csv")
REPORTS_DIR: str = os.path.join(BASE_DIR, "reports", "eda")
FIGURES_DIR: str = os.path.join(REPORTS_DIR, "figures")

# ─── Schema ──────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = ["text", "label"]
VALID_LABELS: set[str] = {
    "normal",
    "hate_speech",
    "insult",
    "harassment",
    "threat",
    "sexually_explicit",
}

# ─── Preprocessing Paths ─────────────────────────────────────────────────────

CLEAN_DATASET_PATH: str = os.path.join(
    BASE_DIR, "dataset", "processed", "final_dataset_clean.csv"
)
PREPROCESSING_REPORTS_DIR: str = os.path.join(BASE_DIR, "reports", "preprocessing")

SLANG_DICT_PATH: str = os.path.join(BASE_DIR, "dataset", "raw", "kamus_singkatan.csv")
ALAY_DICT_PATH: str = os.path.join(
    BASE_DIR, "dataset", "raw", "hatespeech & abusive", "new_kamusalay.csv"
)

# ─── ML Config ───────────────────────────────────────────────────────────────

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

# ─── Feature Engineering Paths ────────────────────────────────────────────────

FEATURE_REPORTS_DIR: str = os.path.join(BASE_DIR, "reports", "feature_engineering")
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
TFIDF_VECTORIZER_PATH: str = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
X_TRAIN_PATH: str = os.path.join(BASE_DIR, "dataset", "processed", "X_train.npz")
X_TEST_PATH: str = os.path.join(BASE_DIR, "dataset", "processed", "X_test.npz")
Y_TRAIN_PATH: str = os.path.join(BASE_DIR, "dataset", "processed", "y_train.csv")
Y_TEST_PATH: str = os.path.join(BASE_DIR, "dataset", "processed", "y_test.csv")

# ─── Training Paths ───────────────────────────────────────────────────────────

TRAINING_REPORTS_DIR: str = os.path.join(BASE_DIR, "reports", "training")
NB_MODEL_PATH: str = os.path.join(MODELS_DIR, "naive_bayes_baseline.joblib")
LR_MODEL_PATH: str = os.path.join(MODELS_DIR, "logistic_regression_baseline.joblib")
SVM_MODEL_PATH: str = os.path.join(MODELS_DIR, "svm_baseline.joblib")

# ─── Tuning Paths ─────────────────────────────────────────────────────────────

TUNING_REPORTS_DIR: str = os.path.join(BASE_DIR, "reports", "tuning")
NB_BEST_MODEL_PATH: str = os.path.join(MODELS_DIR, "naive_bayes_best.joblib")
LR_BEST_MODEL_PATH: str = os.path.join(MODELS_DIR, "logistic_regression_best.joblib")
SVM_BEST_MODEL_PATH: str = os.path.join(MODELS_DIR, "svm_best.joblib")

# ─── Evaluation Paths ─────────────────────────────────────────────────────────

EVALUATION_REPORTS_DIR: str = os.path.join(BASE_DIR, "reports", "evaluation")
EVAL_CLASSIFICATION_REPORTS_DIR: str = os.path.join(EVALUATION_REPORTS_DIR, "classification_reports")
EVAL_CONFUSION_MATRICES_DIR: str = os.path.join(EVALUATION_REPORTS_DIR, "confusion_matrices")
EVAL_ROC_CURVES_DIR: str = os.path.join(EVALUATION_REPORTS_DIR, "roc_curves")
EVAL_FIGURES_DIR: str = os.path.join(EVALUATION_REPORTS_DIR, "figures")
EVAL_COMPARISON_DIR: str = os.path.join(EVALUATION_REPORTS_DIR, "comparison")

# ─── Error Analysis Paths ─────────────────────────────────────────────────────

ERROR_ANALYSIS_REPORTS_DIR: str = os.path.join(BASE_DIR, "reports", "error_analysis")
ERROR_FIGURES_DIR: str = os.path.join(ERROR_ANALYSIS_REPORTS_DIR, "figures")
EVAL_PREDICTIONS_DIR: str = os.path.join(EVALUATION_REPORTS_DIR, "predictions")

# ─── Explainability Paths ─────────────────────────────────────────────────────

EXPLAINABILITY_REPORTS_DIR: str = os.path.join(BASE_DIR, "reports", "explainability")
EXPLAIN_FIGURES_DIR: str = os.path.join(EXPLAINABILITY_REPORTS_DIR, "figures")
EXPLAIN_RAW_DIR: str = os.path.join(EXPLAINABILITY_REPORTS_DIR, "raw")
