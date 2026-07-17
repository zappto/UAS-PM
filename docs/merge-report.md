# Merge Report

**Generated**: 2026-07-15 03:21:49

---

## Merged Datasets

| Dataset | Rows | Status |
|---------|------|--------|
| cyberbullying_cleaned_indo | 2,391 | ✅ Merged |
| hatespeech_abusive | 13,169 | ✅ Merged |
| indotoxic2024 | 28,447 | ✅ Merged |
| **Total** | **44,007** | |

---

## Excluded Datasets

| Dataset | Original Rows | Reason |
|---------|---------------|--------|
| combined_dataset.csv | 2,066 | Sentiment analysis labels (positif/negatif/positive/negative). Not cyberbullying type classification. **Deleted.** |
| re_dataset.csv | 13,169 | Exact duplicate of data.csv (byte-identical). **Deleted.** |
| abusive.csv | 125 | Abusive word list (support file, not training data) |
| new_kamusalay.csv | 15,166 | Slang normalization dictionary (support file) |
| kamus_singkatan.csv | 1,503 | Abbreviation dictionary (support file) |

---

## Merge Statistics

- **Total samples before merge**: 44,007 (across 3 datasets)
- **Total samples after merge**: 44,007
- **Total classes**: 6

### Label Distribution After Merge

| Label | Count | Percentage |
|-------|-------|------------|
| `normal` | 30,576 | 69.48% |
| `hate_speech` | 7,486 | 17.01% |
| `insult` | 3,488 | 7.93% |
| `harassment` | 1,276 | 2.9% |
| `threat` | 1,089 | 2.47% |
| `sexually_explicit` | 92 | 0.21% |

---

## Notes

- Merged dataset has NOT been deduplicated yet (Phase 6).
- Label mapping was applied before merging.
- The `sexually_explicit` class may have very few samples.
