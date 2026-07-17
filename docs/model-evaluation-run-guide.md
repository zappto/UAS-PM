# Model Evaluation Run Guide

## Overview

This guide explains how to execute the model evaluation pipeline for the **Cyberbullying Type Classification** research project.

The best-tuned models are evaluated on the isolated **testing dataset**. The pipeline calculates metrics (Accuracy, Precision, Recall, F1), generates confusion matrix heatmaps (via `matplotlib`), plots ROC curves (One-vs-Rest), and produces a final model comparison table.

---

## 1. Required Python Packages

| Package | Purpose | Version |
|---------|---------|---------|
| scikit-learn | ML metrics, label binarization | 1.3+ |
| scipy | Sparse matrix loading | 1.10+ |
| joblib | Model loading | 1.3+ |
| pandas | Data handling, CSV export | 2.0+ |
| numpy | Array operations | 1.24+ |
| matplotlib | CM Heatmaps & ROC Curves (No seaborn) | 3.7+ |

---

## 2. Installation Commands

```bash
pip install scikit-learn scipy joblib pandas numpy matplotlib
```

### Verify Installation

```bash
python3 -c "
import sklearn; print(f'scikit-learn: {sklearn.__version__}')
import pandas; print(f'pandas: {pandas.__version__}')
import matplotlib; print(f'matplotlib: {matplotlib.__version__}')
print('\n✓ All dependencies installed.')
"
```

---

## 3. Expected Project Structure

```
project/
├── dataset/processed/
│   ├── X_test.npz               ← INPUT
│   └── y_test.csv               ← INPUT
│
├── models/
│   ├── naive_bayes_best.joblib        ← INPUT (from tuning)
│   ├── logistic_regression_best.joblib ← INPUT (from tuning)
│   └── svm_best.joblib                ← INPUT (from tuning)
│
├── src/
│   ├── config/
│   │   └── evaluation_config.py  ← Configuration
│   └── evaluation/
│       ├── validator.py          ← Input validation
│       ├── metrics.py            ← F1, Precision, Recall, Accuracy
│       ├── classification_report.py ← Class-level reports
│       ├── confusion_matrix.py   ← Heatmaps (matplotlib)
│       ├── roc_curve.py          ← OvR ROC plots
│       ├── evaluator.py          ← Single model evaluation
│       ├── model_comparison.py   ← Aggregation & bar charts
│       ├── report_generator.py   ← Markdown summary gen
│       └── persistence.py        ← Save artifacts
│
├── notebooks/
│   └── 06_model_evaluation.ipynb
│
└── reports/evaluation/            ← OUTPUTS DIRECTORY
    ├── evaluation_summary.md
    ├── model_comparison.csv
    ├── metrics.json
    ├── classification_reports/
    ├── confusion_matrices/
    ├── roc_curves/
    └── comparison/
```

---

## 4. Required Input Files

These files must exist before running the evaluation notebook:

| File | Source |
|------|--------|
| `dataset/processed/X_test.npz` | Feature Engineering (notebook 03) |
| `dataset/processed/y_test.csv` | Feature Engineering (notebook 03) |
| `models/naive_bayes_best.joblib` | Hyperparameter Tuning (notebook 05) |
| `models/logistic_regression_best.joblib` | Hyperparameter Tuning (notebook 05) |
| `models/svm_best.joblib` | Hyperparameter Tuning (notebook 05) |

---

## 5. How to Execute — Jupyter Notebook

```bash
cd /home/zapp/Kampus/PM
jupyter notebook notebooks/06_model_evaluation.ipynb
```

Run each cell sequentially (Shift+Enter).

---

## 6. How to Execute — Python Script

```bash
cd /home/zapp/Kampus/PM

python3 -c "
import sys, joblib; sys.path.insert(0, '.')
from src.config.settings import NB_BEST_MODEL_PATH, LR_BEST_MODEL_PATH, SVM_BEST_MODEL_PATH, EVAL_COMPARISON_DIR
from src.config.evaluation_config import EvaluationConfig
from src.utils.load_features import load_features
from src.evaluation.evaluator import evaluate_model
from src.evaluation.model_comparison import generate_comparison_dataframe, generate_all_comparisons
from src.evaluation.report_generator import generate_evaluation_summary
from src.evaluation.persistence import save_evaluation_artifacts

# Load data (only testing)
_, X_test, _, y_test = load_features()

# Load models
nb = joblib.load(NB_BEST_MODEL_PATH)
lr = joblib.load(LR_BEST_MODEL_PATH)
svm = joblib.load(SVM_BEST_MODEL_PATH)

# Evaluate
config = EvaluationConfig()
r_nb = evaluate_model(nb, 'Naive Bayes', X_test, y_test, config)
r_lr = evaluate_model(lr, 'Logistic Regression', X_test, y_test, config)
r_svm = evaluate_model(svm, 'SVM', X_test, y_test, config)
results = [r_nb, r_lr, r_svm]

# Compare & Save
comp_df = generate_comparison_dataframe(results)
generate_all_comparisons(comp_df, EVAL_COMPARISON_DIR)
summary = generate_evaluation_summary(comp_df, results, len(y_test))
save_evaluation_artifacts(results, comp_df, summary, config)

print('\n✓ Evaluation complete. See reports/evaluation/')
"
```

---

## 7. Expected Generated Files

| Directory/File | Format | Description |
|----------------|--------|-------------|
| `evaluation_summary.md` | Markdown | Main report (links to artifacts) |
| `model_comparison.csv` | CSV | NB, LR, SVM sorted by F1 |
| `comparison/` | PNG | Bar charts (F1, Acc, Prec, Rec) |
| `confusion_matrices/` | PNG, CSV | Heatmaps & raw arrays |
| `roc_curves/` | PNG | One-vs-Rest ROC curves per model |
| `classification_reports/`| CSV, MD | Per-class metrics per model |

---

## 8. Troubleshooting

### FileNotFoundError: X_test.npz or models

You skipped a step. Make sure notebook 03 (Features) and notebook 05 (Tuning) have been executed successfully.

### AttributeError: 'SVC' object has no attribute 'predict_proba'

Your SVM was not wrapped in `CalibratedClassifierCV`. Go back to Notebook 05 (Hyperparameter Tuning) and ensure the tuned SVM is saved correctly as a calibrated classifier.

### ImportError: No module named 'src'

Run from the project root directory.

---

## 9. Validation Checklist

After running, verify:

```bash
# 1. Check reports exist
ls -la reports/evaluation/

# 2. Check visualizations
ls -la reports/evaluation/confusion_matrices/
ls -la reports/evaluation/roc_curves/

# 3. View comparison CSV
cat reports/evaluation/model_comparison.csv
```
