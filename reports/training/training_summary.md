# Baseline Model Training Summary

**Generated**: 2026-07-16 15:00:19

---

## Overview

| Metric | Value |
|--------|-------|
| Training Samples | 33,244 |
| Number of Features | 22,695 |
| Models Trained | 3 |
| Random State | 42 |
| Parameter Mode | Default (Baseline) |

---

## Training Results

| Model | Sklearn Class | Training Time (s) | Status |
|-------|---------------|-------------------|--------|
| NaiveBayes | MultinomialNB | 0.0829 | ✓ Trained |
| LogisticRegression | LogisticRegression | 13.1168 | ✓ Trained |
| SVM | CalibratedClassifierCV | 526.2801 | ✓ Trained |

---

## Saved Model Files

| Model | File |
|-------|------|
| Naive Bayes | `models/naive_bayes_baseline.joblib` |
| Logistic Regression | `models/logistic_regression_baseline.joblib` |
| Support Vector Machine | `models/svm_baseline.joblib` |

---

## Notes

- All models use **default parameters** (baseline).
- No hyperparameter tuning was applied.
- No evaluation metrics are included in this report.
- Evaluation will be performed in `05_model_evaluation.ipynb`.
- Hyperparameter tuning will be performed in `05_hyperparameter_tuning.ipynb`.

---

*Report generated automatically by the training pipeline.*
