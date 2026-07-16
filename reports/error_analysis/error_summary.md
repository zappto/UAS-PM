# Error Analysis Summary

**Analysis Date**: 2026-07-16 22:02:43

---

## Overview

| Metric | Value |
|--------|-------|
| Models Analyzed | naive_bayes, logistic_regression, svm |
| Dataset Used | `X_test.npz`, `y_test.csv`, `final_dataset_clean.csv` |
| Test Samples | 8,312.0 |

---

## Error Statistics

| model               |   total_samples |   correct_predictions |   errors | correct_percentage   | error_percentage   |
|:--------------------|----------------:|----------------------:|---------:|:---------------------|:-------------------|
| naive_bayes         |            8312 |                  6564 |     1748 | 78.97%               | 21.03%             |
| logistic_regression |            8312 |                  6556 |     1756 | 78.87%               | 21.13%             |
| svm                 |            8312 |                  6735 |     1577 | 81.03%               | 18.97%             |

---

## Most Difficult Classes

Ranked by average Recall Error Rate (False Negative Rate) across all models.

| class             |   naive_bayes_fn_rate |   logistic_regression_fn_rate |   svm_fn_rate |   avg_recall_error_rate |
|:------------------|----------------------:|------------------------------:|--------------:|------------------------:|
| threat            |                0.9907 |                        0.9395 |        0.9628 |                  0.9643 |
| sexually_explicit |                0.8889 |                        0.8333 |        0.7778 |                  0.8333 |
| insult            |                0.7139 |                        0.6622 |        0.7109 |                  0.6957 |

---

## Most Frequently Confused Classes

Aggregated top pairwise misclassifications across all models.

| true_label   | predicted_label   |   count |
|:-------------|:------------------|--------:|
| hate_speech  | normal            |    1447 |
| insult       | normal            |    1001 |
| normal       | hate_speech       |     675 |
| threat       | normal            |     502 |
| insult       | hate_speech       |     352 |
| harassment   | normal            |     261 |
| normal       | insult            |     206 |
| hate_speech  | insult            |     167 |
| threat       | hate_speech       |      79 |
| harassment   | hate_speech       |      61 |

---

## Generated Files

- **Statistics**: `error_statistics.csv`
- **Class Errors**: `class_error_analysis.csv`, `difficult_classes.csv`
- **Confusions**: `confusion_analysis.csv`
- **Text Patterns**: `frequent_error_words.csv`
- **Raw Data**: `raw/misclassified_samples.csv`
- **Figures**: See `figures/` directory.

---

*Report generated automatically by the error analysis pipeline.*
