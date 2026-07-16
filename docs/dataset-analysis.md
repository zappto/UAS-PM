# Dataset Analysis Report

**Generated**: 2026-07-15 03:21:48

---

## Training Dataset Candidates

| Dataset | Language | Rows | Text Column | Label Column | Classes | Missing Values | Duplicate Rows | Suitable? |
|---------|----------|------|-------------|--------------|---------|----------------|----------------|-----------|
| cyberbullying_cleaned_indo.csv | Indonesian (machine-translated) | 2,400 | `clean_text` | `cyberbullying_type` | 6 | 9 | 0 | **Suitable** |
| hatespeech & abusive/data.csv | Indonesian (native) | 13,169 | `Tweet` | `HS, Abusive, HS_* subcategories` | 2 binary (HS, Abusive) + 7 subcategories → 3 derived classes | 0 | 125 | **Suitable** |
| indotoxic2024_annotated_data_v2_final.csv | Indonesian (native) | 28,448 | `text` | `Multi-annotator binary (toxicity, insults, threat, identity_attack, profanity, sexually_explicit)` | 6 annotation columns → multi-class derived | 1 | 0 | **Partially Suitable** |

---

## Suitability Analysis

### cyberbullying_cleaned_indo.csv

**Status**: Suitable

**Reason**: Has multi-class cyberbullying type labels that directly match the research requirements. Text appears machine-translated from English but labels are ideal (age, ethnicity, gender, religion, other_cyberbullying, not_cyberbullying).

**Label Distribution**:

- `age`: 400
- `ethnicity`: 400
- `gender`: 400
- `not_cyberbullying`: 400
- `other_cyberbullying`: 400
- `religion`: 400

---

### hatespeech & abusive/data.csv

**Status**: Suitable

**Reason**: Native Indonesian tweets with detailed hate speech subcategories (Religion, Race, Physical, Gender, Other). Can be mapped to cyberbullying types. Large dataset (13,169 rows). Has 125 duplicate rows and 146 text duplicates that need cleaning.

**Label Distribution**:

- `normal (derived)`: 5,860
- `hate_speech (derived)`: 5,561
- `insult (derived)`: 1,748

---

### indotoxic2024_annotated_data_v2_final.csv

**Status**: Partially Suitable

**Reason**: Authentic Indonesian text from social media with 28,448 rows. Uses multi-annotator binary labeling across 6 toxicity categories. Requires annotator disagreement resolution and multi-label to single-label conversion. Labels can be mapped to cyberbullying types. Very large dataset but complex transformation needed.

**Label Distribution**:

- `toxicity` (at least 1 annotator positive): 5,273
- `insults` (at least 1 annotator positive): 2,985
- `threat_incitement_to_violence` (at least 1 annotator positive): 1,472
- `identity_attack` (at least 1 annotator positive): 2,776
- `profanity_obscenity` (at least 1 annotator positive): 1,272
- `sexually_explicit` (at least 1 annotator positive): 249

---

## Support Files (Not Training Datasets)

| File | Rows | Description | Status |
|------|------|-------------|--------|
| hatespeech & abusive/abusive.csv | 125 | Abusive word list (125 words). Support file for text analysis, NOT a training dataset. | Not Suitable (Support File) |
| hatespeech & abusive/new_kamusalay.csv | 15,166 | Slang/alay normalization dictionary (15,166 entries). Support file for preprocessing, NOT a training dataset. | Not Suitable (Support File) |
| kamus_singkatan.csv | 1,503 | Indonesian abbreviation dictionary (1,503 entries). Support file for preprocessing, NOT a training dataset. | Not Suitable (Support File) |

---

## Deleted Files

| File | Original Rows | Reason | Status |
|------|---------------|--------|--------|
| combined_dataset.csv (DELETED) | 2,066 | Sentiment analysis dataset with labels: positif, negatif, positive, negative, Bullying, Non-bullying. These are sentiment polarity labels, NOT cyberbullying type labels. Deleted per user approval. | Not Suitable (Deleted) |
| hatespeech & abusive/re_dataset.csv (DELETED) | 13,169 | Exact duplicate of data.csv (byte-for-byte identical). Deleted to avoid confusion. | Not Suitable (Deleted) |

---

## Summary

- **3 datasets** are suitable for training (after transformation)
- **3 files** are support/preprocessing resources (kept for later use)
- **2 files** were deleted (sentiment dataset + duplicate)
