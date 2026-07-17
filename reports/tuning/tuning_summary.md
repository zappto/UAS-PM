# Hyperparameter Tuning Report

**Generated**: 2026-07-16 16:48:41

---

## GridSearchCV Configuration

| Parameter | Value |
|-----------|-------|
| cv | `5` |
| scoring | `f1_macro` |
| verbose | `1` |
| n_jobs | `-1` |
| refit | `True` |
| return_train_score | `True` |
| save_results | `True` |
| random_state | `42` |

---

## Tuning Summary

| Model | Best Score | Candidates | CV Iterations | Duration (s) | Status |
|-------|-----------|------------|---------------|-------------|--------|
| NaiveBayes | 0.4216 | 5 | 25 | 3.44 | ✓ Completed |
| LogisticRegression | 0.4644 | 12 | 60 | 91.86 | ✓ Completed |
| SVM | 0.4645 | 5 | 25 | 1485.58 | ✓ Completed |

---

## 1. NaiveBayes

### Parameter Grid

| Parameter | Search Values |
|-----------|---------------|
| `alpha` | [0.1, 0.5, 1.0, 2.0, 5.0] |

### Best Parameters

| Parameter | Best Value |
|-----------|------------|
| `alpha` | `0.1` |

**Best CV Score (f1_macro)**: `0.421576`

**Search Duration**: 3.44s

**Candidates Tested**: 5

**Total CV Fits**: 25

---

## 2. LogisticRegression

### Parameter Grid

| Parameter | Search Values |
|-----------|---------------|
| `C` | [0.01, 0.1, 1.0, 10.0] |
| `solver` | ['lbfgs'] |
| `max_iter` | [500, 1000, 2000] |

### Best Parameters

| Parameter | Best Value |
|-----------|------------|
| `C` | `10.0` |
| `max_iter` | `500` |
| `solver` | `lbfgs` |

**Best CV Score (f1_macro)**: `0.464396`

**Search Duration**: 91.86s

**Candidates Tested**: 12

**Total CV Fits**: 60

---

## 3. SVM

### Parameter Grid

| Parameter | Search Values |
|-----------|---------------|
| `C` | [0.01, 0.1, 1.0, 10.0, 100.0] |

### Best Parameters

| Parameter | Best Value |
|-----------|------------|
| `C` | `1.0` |

**Best CV Score (f1_macro)**: `0.464451`

**Search Duration**: 1485.58s

**Candidates Tested**: 5

**Total CV Fits**: 25

---

## Notes

- All scores are **cross-validation scores on the training set**.
- The test set was NOT used during tuning.
- No evaluation metrics are included in this report.
- Model evaluation will be performed in `06_model_evaluation.ipynb`.

---

*Report generated automatically by the tuning pipeline.*
