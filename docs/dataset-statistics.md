# Dataset Statistics Report

**Generated**: 2026-07-15 03:21:50

---

## Overview

| Metric | Value |
|--------|-------|
| Total Rows | 41,556 |
| Total Classes | 6 |
| Missing Text Values | 0 |
| Missing Label Values | 0 |
| Duplicate Rows (text-based) | 0 |
| Average Text Length (words) | 34.47 |
| Shortest Text (words) | 1 |
| Longest Text (words) | 880 |
| Vocabulary Size (unique words) | 169,040 |

---

## Samples per Class

| Label | Count | Percentage |
|-------|-------|------------|
| `normal` | 28,462 | 68.49% |
| `hate_speech` | 7,282 | 17.52% |
| `insult` | 3,390 | 8.16% |
| `harassment` | 1,258 | 3.03% |
| `threat` | 1,073 | 2.58% |
| `sexually_explicit` | 91 | 0.22% |

---

## Cleaning Summary

| Step | Rows Removed |
|------|--------------|
| Empty text | 0 |
| Invalid labels | 0 |
| Corrupted text | 0 |
| Duplicate rows | 2451 |
| **Total removed** | **2,451** |
| **Initial count** | 44,007 |
| **Final count** | 41,556 |

---

## Text Length Distribution

| Range (words) | Count |
|---------------|-------|
| 1–5 | 2,471 |
| 6–10 | 7,798 |
| 11–20 | 12,790 |
| 21–50 | 13,733 |
| 51–100 | 1,869 |
| 101–200 | 1,907 |
| 201–∞ | 988 |

---

## Research Readiness Assessment

### Dataset Readiness for ML Pipeline

| Requirement | Status | Notes |
|-------------|--------|-------|
| Minimum 5,000 samples | ✅ Pass | 41,556 samples |
| Multi-class (≥3 classes) | ✅ Pass | 6 classes |
| No missing values | ✅ Pass | text=0, label=0 |
| No duplicate texts | ✅ Pass | 0 duplicates |
| Consistent label format | ✅ Pass | All labels are lowercase snake_case |
| Text column valid | ✅ Pass | All entries are non-empty strings |

---

### Readiness for Specific Algorithms

| Algorithm | Ready? | Notes |
|-----------|--------|-------|
| **TF-IDF** | ✅ Ready | Text data suitable for bag-of-words representation |
| **Naive Bayes** | ✅ Ready | Works well with TF-IDF features, handles multi-class |
| **Logistic Regression** | ✅ Ready | Handles high-dimensional sparse TF-IDF features well |
| **Support Vector Machine** | ✅ Ready | Effective with TF-IDF, good for text classification |

---

### Remaining Considerations

1. **Class Imbalance**: Ratio 312.77:1 (largest/smallest class). Significant imbalance detected. Consider using stratified split, class weights, or oversampling/undersampling.
2. **Text Quality**: Dataset includes machine-translated text (from cyberbullying_cleaned_indo). This may affect model performance on authentic Indonesian text.
3. **Preprocessing Pipeline**: Text still needs lowercase, cleaning, tokenization, stopword removal, and stemming before TF-IDF extraction (as defined in TRD).
4. **Train/Test Split**: Use 80/20 stratified split with random_state=42 as specified in TRD.

---

## Final Dataset Location

```
dataset/processed/final_dataset.csv
dataset/cleaned/final_dataset.csv   (copy per TRD structure)
```

**Schema**: `text` (string), `label` (string)

**Ready for next phase**: ✅ Preprocessing → TF-IDF → Model Training
