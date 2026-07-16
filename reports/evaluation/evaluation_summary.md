# Model Evaluation Summary

**Evaluation Date**: 2026-07-16 20:50:32

---

## Overview

| Metric | Value |
|--------|-------|
| Models Evaluated | 3 |
| Dataset Used | `X_test.npz`, `y_test.csv` |
| Test Samples | 8,312 |
| Primary Metric | F1 Score (Macro) |

---

## Overall Model Comparison

| Model                  |   accuracy |   precision_macro |   recall_macro |   f1_macro |   precision_weighted |   recall_weighted |   f1_weighted |   roc_auc_macro |
|:-----------------------|-----------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|----------------:|
| Support Vector Machine |     0.8103 |            0.7522 |         0.4472 |     0.5042 |               0.7966 |            0.8103 |        0.7847 |          0.8135 |
| Logistic Regression    |     0.7887 |            0.6242 |         0.4374 |     0.4837 |               0.7651 |            0.7887 |        0.7716 |          0.8402 |
| Naive Bayes            |     0.7897 |            0.6126 |         0.3908 |     0.4279 |               0.7716 |            0.7897 |        0.763  |          0.8244 |

[📥 Download full comparison CSV](model_comparison.csv)

---

## Evaluation Visualizations

### Naive Bayes

- **Classification Report**: [`naive_bayes_report.csv`](classification_reports/naive_bayes_report.csv)
- **Confusion Matrix**: [`naive_bayes_cm.png`](confusion_matrices/naive_bayes_cm.png)
- **ROC Curve (OvR)**: [`naive_bayes_roc.png`](roc_curves/naive_bayes_roc.png)

### Logistic Regression

- **Classification Report**: [`logistic_regression_report.csv`](classification_reports/logistic_regression_report.csv)
- **Confusion Matrix**: [`logistic_regression_cm.png`](confusion_matrices/logistic_regression_cm.png)
- **ROC Curve (OvR)**: [`logistic_regression_roc.png`](roc_curves/logistic_regression_roc.png)

### Support Vector Machine

- **Classification Report**: [`support_vector_machine_report.csv`](classification_reports/support_vector_machine_report.csv)
- **Confusion Matrix**: [`support_vector_machine_cm.png`](confusion_matrices/support_vector_machine_cm.png)
- **ROC Curve (OvR)**: [`support_vector_machine_roc.png`](roc_curves/support_vector_machine_roc.png)

---

## Notes

- All metrics calculated on the isolated **testing dataset**.
- ROC Curves use the **One-vs-Rest (OvR)** approach for multi-class classification.
- This report contains objective metrics only. See `07_error_analysis.ipynb` for detailed analysis.

---

*Report generated automatically by the evaluation pipeline.*
