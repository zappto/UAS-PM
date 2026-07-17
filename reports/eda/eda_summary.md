# Exploratory Data Analysis (EDA) Report

**Generated**: 2026-07-15 04:01:02

**Dataset**: `dataset/processed/final_dataset.csv`

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Rows | 41,556 |
| Total Columns | 2 |
| Data Type (`text`) | str |
| Data Type (`label`) | str |
| Memory Usage | 26.25 MB |

---

## 2. Data Quality

### Missing Values

| Column | Missing Count | Missing Percentage |
|--------|---------------|--------------------|
| text | 0 | 0.0% |
| label | 0 | 0.0% |

### Duplicate Analysis

| Metric | Value |
|--------|-------|
| Duplicate Rows | 0 |
| Duplicate Percentage | 0.0% |
| Total Rows | 41,556 |

### Data Quality Issues

| Issue | Count |
|-------|-------|
| Missing Text | ✓ |
| Missing Label | ✓ |
| Empty Text (whitespace only) | ✓ |
| Very Short Text (< 3 words) | ⚠ 150 |
| Extremely Long Text (> 500 words) | ⚠ 135 |
| Invalid Labels | ✓ |
| Duplicate Rows | ✓ |

---

## 3. Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| `normal` | 28,462 | 68.49% |
| `hate_speech` | 7,282 | 17.52% |
| `insult` | 3,390 | 8.16% |
| `harassment` | 1,258 | 3.03% |
| `threat` | 1,073 | 2.58% |
| `sexually_explicit` | 91 | 0.22% |

### Class Imbalance Assessment

| Metric | Value |
|--------|-------|
| Imbalance Ratio | 312.77:1 |
| Status | **Severely Imbalanced** |
| Largest Class | `normal` (28,462) |
| Smallest Class | `sexually_explicit` (91) |

> The dataset is severely imbalanced. The ratio between the largest class ('normal': 28,462) and the smallest class ('sexually_explicit': 91) is 312.77:1. Consider oversampling, undersampling, class weights, or SMOTE. Use stratified split and prioritize F1-Score over accuracy.

---

## 4. Text Statistics

### Word Count

| Statistic | Value |
|-----------|-------|
| Mean | 34.47 |
| Median | 18.0 |
| Mode | 10 |
| Std | 58.45 |
| Min | 1 |
| Max | 880 |

### Character Count

| Statistic | Value |
|-----------|-------|
| Mean | 240.53 |
| Median | 121.0 |
| Mode | 254 |
| Std | 426.48 |
| Min | 2 |
| Max | 4998 |

### Vocabulary

| Metric | Value |
|--------|-------|
| Unique Words (Vocabulary Size) | 200,199 |
| Total Words | 1,432,248 |

---

## 5. Research Findings

- Dataset contains 41,556 samples across 6 classes.
- Class imbalance status: **Severely Imbalanced** (ratio 312.77:1).
- The dominant class is `normal` with 28,462 samples (68.49%).
- The smallest class is `sexually_explicit` with 91 samples (0.22%).
- Average text length is 34.47 words (median: 18.0, std: 58.45).
- Vocabulary size is 200,199 unique words from 1,432,248 total words.
- Dataset contains no duplicate rows.
- There are 150 very short texts (< 3 words) that may need attention during preprocessing.
- There are 135 extremely long texts (> 500 words) that may affect TF-IDF performance.

---

## 6. Recommendations before Preprocessing

1. **Handle class imbalance**: Consider applying class weights, oversampling (SMOTE), or undersampling to address the severe imbalance (ratio 312.77:1).
2. **Text cleaning**: Remove URLs, mentions, hashtags, emojis, HTML tags, punctuation, numbers, and extra whitespace as specified in the DRD.
3. **Text preprocessing**: Apply case folding → tokenization → stopword removal → stemming (Sastrawi) → join tokens.
4. **Review short texts**: 150 texts have fewer than 3 words. Verify they contain meaningful content before TF-IDF extraction.
5. **TF-IDF tuning**: With a vocabulary of 200,199 unique words, consider setting max_features and min_df/max_df parameters to reduce dimensionality.
6. **Data split**: Use 80/20 stratified split with random_state=42 as defined in the TRD.

---

*Report generated automatically by the EDA pipeline.*
