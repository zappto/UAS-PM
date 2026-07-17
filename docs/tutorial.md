# Tutorial — Dataset Preparation Pipeline

## Overview

This guide explains how to run the dataset preparation pipeline for the **Cyberbullying Type Classification** research project.

The pipeline processes raw datasets from Kaggle, standardizes schemas, maps labels, merges compatible datasets, cleans duplicates, and validates the final dataset for ML training.

---

## Prerequisites

### System Requirements

- Python 3.10+
- Operating System: Linux / macOS / Windows

### Python Dependencies

```bash
# Install required packages
pip install pandas numpy
```

> **Note**: The pipeline script (`src/prepare_dataset.py`) uses only Python's standard library (`csv`, `os`, `re`, `ast`, `collections`). Pandas is optional — only needed if you want to inspect the data interactively in notebooks.

---

## Project Structure

```
project/
├── dataset/
│   ├── raw/                  ← Raw datasets from Kaggle (input)
│   ├── interim/              ← Standardized + merged datasets
│   ├── cleaned/              ← Cleaned dataset (per TRD structure)
│   └── processed/            ← Final dataset ready for ML
│
├── docs/                     ← Generated documentation
│   ├── context.md
│   ├── prd.md
│   ├── trd.md
│   ├── drd.md
│   ├── dataset-analysis.md   ← Phase 1 output
│   ├── label-analysis.md     ← Phase 3 output
│   ├── merge-report.md       ← Phase 5 output
│   └── dataset-statistics.md ← Phase 7+8 output
│
├── src/
│   └── prepare_dataset.py    ← Main pipeline script
│
├── notebooks/                ← Jupyter notebooks (next phases)
├── models/                   ← Trained models (next phases)
├── streamlit/                ← Streamlit app (next phases)
└── reports/                  ← Reports (next phases)
```

---

## Step-by-Step Guide

### Step 1 — Place Raw Datasets

Ensure your raw datasets are in the `dataset/raw/` directory:

```
dataset/raw/
├── cyberbullying_cleaned_indo.csv
├── indotoxic2024_annotated_data_v2_final.csv
├── kamus_singkatan.csv
└── hatespeech & abusive/
    ├── data.csv
    ├── abusive.csv
    └── new_kamusalay.csv
```

These datasets should be downloaded from Kaggle. The following datasets were excluded (not needed):

- `combined_dataset.csv` — Sentiment analysis, not cyberbullying types
- `re_dataset.csv` — Duplicate of `data.csv`

---

### Step 2 — Run the Pipeline

```bash
# From the project root directory
python3 src/prepare_dataset.py
```

The script will execute all 8 phases automatically:

| Phase | Description | Output |
|-------|-------------|--------|
| 1 | Dataset Inspection | `docs/dataset-analysis.md` |
| 2 | Schema Standardization | `dataset/interim/*_standardized.csv` |
| 3 | Label Analysis | `docs/label-analysis.md` |
| 4 | Label Mapping | `dataset/interim/label_mapping.csv` |
| 5 | Dataset Merge | `dataset/interim/merged_dataset.csv`, `docs/merge-report.md` |
| 6 | Dataset Cleaning | `dataset/processed/final_dataset.csv` |
| 7 | Dataset Validation | `docs/dataset-statistics.md` |
| 8 | Research Readiness | Appended to `docs/dataset-statistics.md` |

---

### Step 3 — Verify Output

After running, verify the final dataset:

```bash
# Check file exists and row count
wc -l dataset/processed/final_dataset.csv

# Preview the first few rows
head -10 dataset/processed/final_dataset.csv

# Check label distribution
python3 -c "
import csv
from collections import Counter
with open('dataset/processed/final_dataset.csv') as f:
    reader = csv.DictReader(f)
    labels = Counter(row['label'] for row in reader)
for label, count in labels.most_common():
    print(f'{label}: {count}')
"
```

Expected output:

```
text,label
normal: ~28,000+
hate_speech: ~7,000+
insult: ~3,000+
harassment: ~1,200+
threat: ~1,000+
sexually_explicit: ~90+
```

---

### Step 4 — Review Documentation

Check the generated reports:

1. **`docs/dataset-analysis.md`** — Suitability analysis for each dataset
2. **`docs/label-analysis.md`** — Label mapping justifications
3. **`docs/merge-report.md`** — Merge statistics and excluded datasets
4. **`docs/dataset-statistics.md`** — Final dataset statistics + research readiness

---

## Final Dataset Schema

The final dataset (`dataset/processed/final_dataset.csv`) has exactly 2 columns:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Indonesian text (comment/tweet/post) |
| `label` | string | Cyberbullying type classification label |

### Label Classes

| Label | Description |
|-------|-------------|
| `normal` | Not cyberbullying |
| `insult` | Insulting/demeaning language |
| `hate_speech` | Identity-based hate speech (religion, race, ethnicity) |
| `threat` | Threats or incitement to violence |
| `harassment` | General harassment (age, gender, profanity-based) |
| `sexually_explicit` | Sexually explicit content |

---

## Next Steps

After dataset preparation is complete, proceed to:

1. **Notebook 01** — Exploratory Data Analysis (`notebooks/01_dataset_analysis.ipynb`)
2. **Notebook 02** — Text Preprocessing (`notebooks/02_preprocessing.ipynb`)
3. **Notebook 03** — TF-IDF Feature Engineering (`notebooks/03_feature_engineering.ipynb`)
4. **Notebook 04** — Model Training: NB, LR, SVM (`notebooks/04_model_training.ipynb`)
5. **Notebook 05** — Model Evaluation (`notebooks/05_model_evaluation.ipynb`)
6. **Notebook 06** — Explainability with SHAP (`notebooks/06_explainability.ipynb`)

---

## Troubleshooting

### "FileNotFoundError: dataset/raw/..."

Ensure the raw datasets are placed correctly in `dataset/raw/`. Check filenames match exactly (case-sensitive on Linux).

### "ModuleNotFoundError: No module named 'pandas'"

The pipeline script does NOT require pandas. If you want to use pandas for interactive analysis in notebooks:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk Sastrawi
```

### Large dataset takes too long

The `indotoxic2024` dataset has ~28,000 rows with multi-annotator labels that need parsing. The pipeline may take 30–60 seconds to complete on modest hardware.

---

## Data Sources

| Dataset | Kaggle Source | License |
|---------|---------------|---------|
| cyberbullying_cleaned_indo.csv | Kaggle Public Dataset | Public / Academic Use |
| hatespeech & abusive/data.csv | Kaggle Public Dataset | Public / Academic Use |
| indotoxic2024_annotated_data_v2_final.csv | Kaggle Public Dataset | Public / Academic Use |

---

## Contact

For issues with the pipeline, check the generated documentation in `docs/` for detailed analysis and justifications for every preprocessing decision.
