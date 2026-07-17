# Hyperparameter Tuning Run Guide

## Overview

This guide explains how to execute the hyperparameter tuning pipeline for the **Cyberbullying Type Classification** research project.

Three models are tuned independently using **GridSearchCV** with 5-fold stratified cross-validation:

1. Multinomial Naive Bayes → alpha
2. Logistic Regression → C, solver, max_iter
3. Linear SVM → C

Scoring metric: **f1_macro**

---

## 1. Required Python Packages

| Package | Purpose | Version |
|---------|---------|---------|
| scikit-learn | GridSearchCV, ML algorithms | 1.3+ |
| scipy | Sparse matrix loading | 1.10+ |
| joblib | Model serialization | 1.3+ |
| pandas | Data handling, CSV export | 2.0+ |
| numpy | Array operations | 1.24+ |

---

## 2. Installation Commands

```bash
pip install scikit-learn scipy joblib pandas numpy
```

### Verify Installation

```bash
python3 -c "
import sklearn; print(f'scikit-learn: {sklearn.__version__}')
import scipy; print(f'scipy: {scipy.__version__}')
import joblib; print(f'joblib: {joblib.__version__}')
import pandas; print(f'pandas: {pandas.__version__}')
print('\n✓ All dependencies installed.')
"
```

---

## 3. Expected Project Structure

```
project/
├── dataset/processed/
│   ├── X_train.npz              ← INPUT
│   ├── X_test.npz               ← INPUT
│   ├── y_train.csv              ← INPUT
│   └── y_test.csv               ← INPUT
│
├── models/
│   ├── tfidf_vectorizer.joblib  ← INPUT
│   ├── naive_bayes_baseline.joblib    ← INPUT (from training)
│   ├── logistic_regression_baseline.joblib ← INPUT
│   ├── svm_baseline.joblib            ← INPUT
│   ├── naive_bayes_best.joblib        ← OUTPUT (generated)
│   ├── logistic_regression_best.joblib ← OUTPUT (generated)
│   └── svm_best.joblib                ← OUTPUT (generated)
│
├── src/
│   ├── config/
│   │   └── tuning_config.py      ← Configuration
│   └── tuning/
│       ├── parameter_grid.py     ← Param grids
│       ├── grid_search.py        ← GridSearchCV wrapper
│       ├── tuner.py              ← Tuning orchestrator
│       ├── model_selector.py     ← Base estimator factory
│       ├── persistence.py        ← Save best models
│       ├── report_generator.py   ← Report generation
│       └── validator.py          ← Input validation
│
├── notebooks/
│   └── 05_hyperparameter_tuning.ipynb
│
└── reports/tuning/
    ├── tuning_summary.md          ← OUTPUT (generated)
    ├── tuning_statistics.csv      ← OUTPUT (generated)
    ├── best_parameters.json       ← OUTPUT (generated)
    └── cv_results.csv             ← OUTPUT (generated)
```

---

## 4. Required Input Files

These files must exist before running the tuning notebook:

| File | Source |
|------|--------|
| `dataset/processed/X_train.npz` | Feature Engineering (notebook 03) |
| `dataset/processed/X_test.npz` | Feature Engineering (notebook 03) |
| `dataset/processed/y_train.csv` | Feature Engineering (notebook 03) |
| `dataset/processed/y_test.csv` | Feature Engineering (notebook 03) |
| `models/tfidf_vectorizer.joblib` | Feature Engineering (notebook 03) |

---

## 5. How to Execute — Jupyter Notebook

```bash
cd /home/zapp/Kampus/PM
jupyter notebook notebooks/05_hyperparameter_tuning.ipynb
```

Run each cell sequentially (Shift+Enter).

⚠️ **Expected runtime**: SVM tuning may take 5–20 minutes depending on CPU.

---

## 6. How to Execute — Python Script

```bash
cd /home/zapp/Kampus/PM

python3 -c "
import sys, os
sys.path.insert(0, '.')

from src.config.settings import (
    TUNING_REPORTS_DIR, NB_BEST_MODEL_PATH,
    LR_BEST_MODEL_PATH, SVM_BEST_MODEL_PATH,
)
from src.config.tuning_config import TuningConfig
from src.utils.load_features import load_features
from src.tuning.tuner import tune_all_models
from src.tuning.persistence import save_best_model, save_best_parameters, save_cv_results
from src.tuning.report_generator import (
    generate_tuning_report, export_tuning_statistics, save_tuning_report,
)

# Load features
X_train, X_test, y_train, y_test = load_features()

# Tune all models
config = TuningConfig()
results = tune_all_models(X_train, y_train, config)

# Save best models
model_paths = {
    'naive_bayes': NB_BEST_MODEL_PATH,
    'logistic_regression': LR_BEST_MODEL_PATH,
    'svm': SVM_BEST_MODEL_PATH,
}
for r in results:
    save_best_model(r['best_model'], model_paths[r['model_key']], overwrite=True)

# Save reports
all_best_params = {r['model_key']: r['best_params'] for r in results}
save_best_parameters(all_best_params, os.path.join(TUNING_REPORTS_DIR, 'best_parameters.json'), overwrite=True)
save_cv_results(results, os.path.join(TUNING_REPORTS_DIR, 'cv_results.csv'), overwrite=True)

report = generate_tuning_report(results, config)
save_tuning_report(report, os.path.join(TUNING_REPORTS_DIR, 'tuning_summary.md'))
export_tuning_statistics(results, os.path.join(TUNING_REPORTS_DIR, 'tuning_statistics.csv'))

print('\n✓ Hyperparameter tuning complete.')
"
```

---

## 7. Expected Generated Files

| File | Format | Description |
|------|--------|-------------|
| `models/naive_bayes_best.joblib` | Joblib | Best tuned MultinomialNB |
| `models/logistic_regression_best.joblib` | Joblib | Best tuned LogisticRegression |
| `models/svm_best.joblib` | Joblib | Best tuned SVC + CalibratedClassifierCV |
| `reports/tuning/tuning_summary.md` | Markdown | Tuning report (no eval metrics) |
| `reports/tuning/tuning_statistics.csv` | CSV | Per-model tuning statistics |
| `reports/tuning/best_parameters.json` | JSON | Best params for all 3 models |
| `reports/tuning/cv_results.csv` | CSV | Full CV results for all folds |

---

## 8. Troubleshooting

### FileNotFoundError: X_train.npz not found

Run the feature engineering pipeline first (notebook 03).

### SVM tuning is very slow

Linear SVM with GridSearchCV on 5 C values × 5 folds = 25 fits. Each fit trains on ~77K samples. Expected: 5–20 minutes.

### ConvergenceWarning during LR tuning

Normal for some (C, solver, max_iter) combinations. GridSearchCV will still report the best converging configuration.

### ImportError: No module named 'src'

Run from the project root:

```bash
cd /home/zapp/Kampus/PM
```

---

## 9. Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: X_train.npz` | Features missing | Run notebook 03 first |
| `ValueError: Parameter grid is empty` | Bad param grid | Check parameter_grid.py |
| `FileExistsError` | Output file exists | Set `overwrite=True` |
| `ModuleNotFoundError` | Package not installed | `pip install scikit-learn` |
| `MemoryError` | Too many candidates | Reduce param grid size |

---

## 10. Validation Checklist

After running, verify:

```bash
# 1. Check best model files exist
ls -la models/naive_bayes_best.joblib
ls -la models/logistic_regression_best.joblib
ls -la models/svm_best.joblib

# 2. Check report files
ls -la reports/tuning/

# 3. Verify best parameters JSON
cat reports/tuning/best_parameters.json

# 4. Verify SVM model supports predict_proba
python3 -c "
import sys; sys.path.insert(0, '.')
import joblib
svm = joblib.load('models/svm_best.joblib')
print(f'SVM type: {type(svm).__name__}')
print(f'Has predict_proba: {hasattr(svm, \"predict_proba\")}')
print('✓ SVM validation passed.')
"
```
