# Model Training Run Guide

## Overview

This guide explains how to execute the baseline model training pipeline for the **Cyberbullying Type Classification** research project.

Three models are trained with **default parameters** to establish a research baseline:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Support Vector Machine (Linear)

No hyperparameter tuning. No evaluation metrics.

---

## 1. Required Python Packages

| Package | Purpose | Version |
|---------|---------|---------|
| scikit-learn | ML algorithms | 1.3+ |
| scipy | Sparse matrix loading | 1.10+ |
| joblib | Model serialization | 1.3+ |
| pandas | Label loading | 2.0+ |
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
│   ├── X_train.npz         ← INPUT (from feature engineering)
│   ├── X_test.npz           ← INPUT
│   ├── y_train.csv           ← INPUT
│   └── y_test.csv            ← INPUT
│
├── models/
│   ├── tfidf_vectorizer.joblib ← INPUT
│   ├── naive_bayes_baseline.joblib    ← OUTPUT (generated)
│   ├── logistic_regression_baseline.joblib ← OUTPUT (generated)
│   └── svm_baseline.joblib            ← OUTPUT (generated)
│
├── src/
│   ├── config/
│   │   ├── settings.py
│   │   └── model_config.py     ← Configuration
│   ├── training/
│   │   ├── base_model.py        ← Abstract base class
│   │   ├── naive_bayes.py       ← NB wrapper
│   │   ├── logistic_regression.py ← LR wrapper
│   │   ├── svm.py               ← SVM wrapper
│   │   ├── model_factory.py     ← Factory pattern
│   │   ├── model_registry.py    ← Model registry
│   │   ├── trainer.py           ← Orchestrator
│   │   └── persistence.py       ← Save/load
│   └── utils/
│       └── load_features.py     ← Feature loader
│
├── notebooks/
│   └── 04_model_training.ipynb
│
└── reports/training/
    ├── training_summary.md      ← OUTPUT (generated)
    └── training_statistics.csv  ← OUTPUT (generated)
```

---

## 4. Required Input Files

These files must exist before running the training notebook:

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
jupyter notebook notebooks/04_model_training.ipynb
```

Run each cell sequentially (Shift+Enter).

---

## 6. How to Execute — Python Script

```bash
cd /home/zapp/Kampus/PM

python3 -c "
import sys, os
sys.path.insert(0, '.')

from src.config.settings import TRAINING_REPORTS_DIR
from src.config.model_config import ModelConfig
from src.utils.load_features import load_features
from src.training.trainer import (
    train_all_models,
    generate_training_report, export_training_statistics, save_training_report,
)

# Load features
X_train, X_test, y_train, y_test = load_features()

# Train all baseline models
config = ModelConfig(overwrite=True)
trained_models = train_all_models(X_train, y_train, config)

# Generate reports
report = generate_training_report(trained_models, X_train.shape[0], X_train.shape[1], config)
save_training_report(report, os.path.join(TRAINING_REPORTS_DIR, 'training_summary.md'))
export_training_statistics(trained_models, X_train.shape[0], X_train.shape[1],
                          os.path.join(TRAINING_REPORTS_DIR, 'training_statistics.csv'))
print('\n✓ Baseline training complete.')
"
```

---

## 7. Expected Generated Files

| File | Format | Description |
|------|--------|-------------|
| `models/naive_bayes_baseline.joblib` | Joblib | Trained MultinomialNB |
| `models/logistic_regression_baseline.joblib` | Joblib | Trained LogisticRegression |
| `models/svm_baseline.joblib` | Joblib | Trained SVC (linear, probability=True) |
| `reports/training/training_summary.md` | Markdown | Training report (no metrics) |
| `reports/training/training_statistics.csv` | CSV | Training timing data |

---

## 8. Troubleshooting

### FileNotFoundError: X_train.npz not found

Run the feature engineering pipeline first (notebook 03):

```bash
jupyter notebook notebooks/03_feature_engineering.ipynb
```

### ConvergenceWarning: LogisticRegression

The baseline uses `max_iter=1000` which should be sufficient. If it still warns, this is normal for baseline and will be addressed during hyperparameter tuning.

### SVM training is very slow

Linear SVM with `probability=True` uses Platt scaling (5-fold CV internally), which is slower than without. On 40K+ samples, expect 2–10 minutes. This is expected.

### ImportError: No module named 'src'

Run from the project root:

```bash
cd /home/zapp/Kampus/PM
```

---

## 9. Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: X_train.npz` | Feature files missing | Run notebook 03 first |
| `ValueError: Feature dimension mismatch` | Corrupted files | Re-run notebook 03 |
| `RuntimeError: build() must be called` | Using BaseModel directly | Use `create_model()` factory |
| `FileExistsError` | Model file exists | Set `overwrite=True` in config |
| `ModuleNotFoundError` | Package not installed | `pip install scikit-learn` |

---

## 10. Validation Checklist

After running, verify:

```bash
# 1. Check model files exist
ls -la models/naive_bayes_baseline.joblib
ls -la models/logistic_regression_baseline.joblib
ls -la models/svm_baseline.joblib

# 2. Check report files
ls -la reports/training/

# 3. Verify models can be loaded
python3 -c "
import sys; sys.path.insert(0, '.')
from src.training.persistence import load_model
nb = load_model('models/naive_bayes_baseline.joblib')
lr = load_model('models/logistic_regression_baseline.joblib')
svm = load_model('models/svm_baseline.joblib')
print('✓ All 3 models loaded successfully.')
"
```
