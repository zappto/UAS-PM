# Feature Engineering Report

**Generated**: 2026-07-15 23:10:52

---

## 1. Dataset

| Metric | Value |
|--------|-------|
| Input File | `dataset/processed/final_dataset_clean.csv` |
| Text Column | `text_clean` |
| Training Samples | 33,244 |
| Testing Samples | 8,312 |
| Total Samples | 41,556 |

---

## 2. Train/Test Split

| Parameter | Value |
|-----------|-------|
| Test Size | 0.2 |
| Random State | 42 |
| Stratified | True |

### Training Label Distribution

| Label | Count |
|-------|-------|
| `normal` | 22,769 |
| `hate_speech` | 5,826 |
| `insult` | 2,712 |
| `harassment` | 1,006 |
| `threat` | 858 |
| `sexually_explicit` | 73 |

### Testing Label Distribution

| Label | Count |
|-------|-------|
| `normal` | 5,693 |
| `hate_speech` | 1,456 |
| `insult` | 678 |
| `harassment` | 252 |
| `threat` | 215 |
| `sexually_explicit` | 18 |

---

## 3. TF-IDF Configuration

| Parameter | Value |
|-----------|-------|
| max_features | `None` |
| min_df | `2` |
| max_df | `0.95` |
| ngram_range | `(1, 1)` |
| norm | `l2` |
| use_idf | `True` |
| smooth_idf | `True` |
| sublinear_tf | `True` |
| analyzer | `word` |
| lowercase | `False` |
| token_pattern | `(?u)\b\w+\b` |

---

## 4. Feature Statistics

| Metric | Value |
|--------|-------|
| Vocabulary Size | 22,695 |
| Number of Features | 22,695 |
| Training Matrix Shape | (33244, 22695) |
| Testing Matrix Shape | (8312, 22695) |
| Matrix Density | 0.000661 |
| Matrix Sparsity | 0.999339 |
| Avg Features per Document | 14.99 |
| Min Features per Document | 0 |
| Max Features per Document | 550 |

---

## 5. Top TF-IDF Terms

| Rank | Term | Avg TF-IDF |
|------|------|------------|
| 1 | `israel` | 0.024755 |
| 2 | `gila` | 0.021286 |
| 3 | `china` | 0.014537 |
| 4 | `orang` | 0.014362 |
| 5 | `x` | 0.012996 |
| 6 | `indonesia` | 0.012417 |
| 7 | `palestina` | 0.011617 |
| 8 | `gaza` | 0.010399 |
| 9 | `serang` | 0.007899 |
| 10 | `dukung` | 0.007302 |
| 11 | `presiden` | 0.007209 |
| 12 | `jokowi` | 0.007145 |
| 13 | `hamas` | 0.007124 |
| 14 | `negara` | 0.007015 |
| 15 | `f` | 0.006869 |
| 16 | `anak` | 0.006869 |
| 17 | `xf` | 0.006521 |
| 18 | `n` | 0.006384 |
| 19 | `warga` | 0.006383 |
| 20 | `islam` | 0.006363 |

---

## 6. Validation

**Status**: ✓ PASS

| Check | Result |
|-------|--------|
| vocabulary_not_empty | ✓ |
| X_train_not_empty | ✓ |
| X_test_not_empty | ✓ |
| feature_dims_match | ✓ |
| train_labels_match | ✓ |
| test_labels_match | ✓ |
| label_sets_consistent | ✓ |
| no_nan_values | ✓ |
| no_inf_values | ✓ |
| no_missing_labels | ✓ |

---

## 7. Recommendations for Model Training

1. Use the saved `tfidf_vectorizer.joblib` to transform new text.
2. Use stratified split to maintain class distribution.
3. Evaluate using F1-Score (macro/weighted) as primary metric.
4. Feature dimensionality is 22,695 — manageable for NB, LR, and SVM.
5. Matrix sparsity is 0.9993 — sparse-aware algorithms (SVM, LR) should perform well.
6. Consider class weights for handling class imbalance.

---

*Report generated automatically by the feature engineering pipeline.*
