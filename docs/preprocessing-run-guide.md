# Preprocessing Run Guide

## Overview

This guide explains how to execute the text preprocessing pipeline for the **Cyberbullying Type Classification** research project.

The pipeline applies the following steps (per TRD):

```
Raw Text → Case Folding → Text Cleaning → Tokenization → Stopword Removal → Stemming → Join Tokens
```

---

## 1. Required Python Packages

| Package | Purpose | Version |
|---------|---------|---------|
| pandas | Data processing | 2.0+ |
| numpy | Numerical operations | 1.24+ |
| nltk | Indonesian stopwords | 3.8+ |
| Sastrawi | Indonesian stemming | 1.0.1+ |

---

## 2. Installation Commands

```bash
# Install required packages
pip install pandas numpy nltk Sastrawi

# Download NLTK Indonesian stopwords
python3 -c "import nltk; nltk.download('stopwords')"
```

### Verify Installation

```bash
python3 -c "
import pandas; print(f'pandas: {pandas.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
import nltk; print(f'nltk: {nltk.__version__}')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory; print('Sastrawi: OK')
from nltk.corpus import stopwords; print(f'NLTK stopwords: {len(stopwords.words(\"indonesian\"))} words')
print('\n✓ All dependencies installed.')
"
```

---

## 3. Project Structure

```
project/
├── dataset/
│   └── processed/
│       ├── final_dataset.csv          ← INPUT
│       └── final_dataset_clean.csv    ← OUTPUT (generated)
│
├── src/
│   ├── config/
│   │   ├── settings.py
│   │   └── preprocessing_config.py    ← Configuration
│   ├── data/
│   │   └── load_dataset.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── case_folding.py            ← Step 1
│   │   ├── text_cleaning.py           ← Step 2
│   │   ├── tokenization.py            ← Step 3
│   │   ├── stopword_removal.py        ← Step 4
│   │   ├── stemming.py                ← Step 5
│   │   ├── pipeline.py                ← Orchestrator
│   │   ├── validator.py               ← Validation
│   │   └── report.py                  ← Report generation
│   └── utils/
│       └── save_dataset.py
│
├── notebooks/
│   └── 02_preprocessing.ipynb          ← Notebook
│
└── reports/
    └── preprocessing/
        ├── preprocessing_summary.md    ← OUTPUT (generated)
        └── preprocessing_statistics.csv ← OUTPUT (generated)
```

---

## 4. How to Execute — Jupyter Notebook

```bash
# Navigate to the project root
cd /home/zapp/Kampus/PM

# Launch Jupyter
jupyter notebook notebooks/02_preprocessing.ipynb
```

Then run each cell sequentially (Shift+Enter).

---

## 5. How to Execute — Python Script

You can also run the pipeline directly from the command line:

```bash
cd /home/zapp/Kampus/PM

python3 -c "
import sys, os
sys.path.insert(0, '.')

from src.data.load_dataset import load_dataset
from src.config.preprocessing_config import PreprocessingConfig
from src.preprocessing.validator import validate_dataset
from src.preprocessing.pipeline import preprocess_dataframe, get_example_transformations
from src.preprocessing.report import generate_preprocessing_report, export_preprocessing_statistics, save_report
from src.utils.save_dataset import save_dataset
from src.config.settings import CLEAN_DATASET_PATH, PREPROCESSING_REPORTS_DIR

# Load
df = load_dataset()

# Validate
validation = validate_dataset(df)
print(validation['summary'])

# Configure
config = PreprocessingConfig()

# Process
def progress(cur, total):
    print(f'  {cur:,}/{total:,} ({cur/total*100:.1f}%)', end='\r')

df_processed, stats = preprocess_dataframe(df, config, progress_callback=progress)

# Save
save_dataset(df_processed[['text', 'label', 'text_clean']], CLEAN_DATASET_PATH, overwrite=True)

# Report
examples = get_example_transformations(df_processed, n=5)
report = generate_preprocessing_report(
    total_rows=stats['total_rows'],
    processed_rows=stats['processed_rows'],
    removed_empty=stats.get('empty_after_preprocessing', 0),
    removed_duplicates=0,
    avg_length_before=stats['avg_word_count_before'],
    avg_length_after=stats['avg_word_count_after'],
    examples=examples,
    processing_time=stats['processing_time_seconds'],
    config=config,
)
save_report(report, os.path.join(PREPROCESSING_REPORTS_DIR, 'preprocessing_summary.md'))
export_preprocessing_statistics(stats, os.path.join(PREPROCESSING_REPORTS_DIR, 'preprocessing_statistics.csv'))
print('\n✓ Preprocessing complete.')
"
```

---

## 6. Expected Input Files

| File | Description |
|------|-------------|
| `dataset/processed/final_dataset.csv` | Merged and deduplicated dataset with `text` and `label` columns |

---

## 7. Expected Output Files

| File | Description |
|------|-------------|
| `dataset/processed/final_dataset_clean.csv` | Preprocessed dataset with `text`, `label`, and `text_clean` columns |
| `reports/preprocessing/preprocessing_summary.md` | Markdown report with overview, steps, examples, and notes |
| `reports/preprocessing/preprocessing_statistics.csv` | CSV with processing metrics |

---

## 8. Troubleshooting

### ModuleNotFoundError: No module named 'Sastrawi'

```bash
pip install Sastrawi
```

### LookupError: NLTK stopwords not found

```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

### FileNotFoundError: Dataset not found

Ensure `dataset/processed/final_dataset.csv` exists. Run the dataset preparation pipeline first:

```bash
python3 src/prepare_dataset.py
```

### ImportError: No module named 'src'

Make sure you are running from the project root directory:

```bash
cd /home/zapp/Kampus/PM
```

### Preprocessing is very slow

Stemming with Sastrawi is the bottleneck (~5–15 minutes for 41K rows). To speed up testing:

```python
config = PreprocessingConfig(enable_stemming=False)
```

---

## 9. Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError` | Dataset CSV missing | Run `prepare_dataset.py` first |
| `ValueError: Missing required columns` | CSV has wrong schema | Check CSV has `text` and `label` columns |
| `ImportError: Sastrawi` | Package not installed | `pip install Sastrawi` |
| `LookupError: stopwords` | NLTK data missing | `nltk.download('stopwords')` |
| `FileExistsError` | Output file exists | Set `overwrite=True` in `save_dataset()` |

---

## 10. Verification Steps

After running the pipeline, verify the outputs:

```bash
# 1. Check output file exists
ls -la dataset/processed/final_dataset_clean.csv

# 2. Check row count matches input
wc -l dataset/processed/final_dataset.csv
wc -l dataset/processed/final_dataset_clean.csv

# 3. Check columns
head -1 dataset/processed/final_dataset_clean.csv
# Expected: text,label,text_clean

# 4. Check sample output
head -3 dataset/processed/final_dataset_clean.csv

# 5. Check reports exist
ls -la reports/preprocessing/

# 6. Quick validation
python3 -c "
import pandas as pd
df = pd.read_csv('dataset/processed/final_dataset_clean.csv')
print(f'Rows: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(f'Missing text_clean: {df[\"text_clean\"].isnull().sum()}')
print(f'Empty text_clean: {(df[\"text_clean\"].astype(str).str.strip() == \"\").sum()}')
print('✓ Verification passed.')
"
```

---

## Configuration

To customize preprocessing, modify the `PreprocessingConfig` in the notebook:

```python
config = PreprocessingConfig(
    enable_case_folding=True,       # Lowercase + Unicode
    enable_url_removal=True,        # Remove URLs
    enable_html_removal=True,       # Remove HTML tags
    enable_mention_removal=True,    # Remove @mentions
    enable_hashtag_removal=True,    # Remove #hashtags
    enable_emoji_removal=True,      # Remove emojis
    enable_punctuation_removal=True,# Remove punctuation
    enable_number_removal=True,     # Remove numbers
    enable_stopword_removal=True,   # Remove Indonesian stopwords
    enable_stemming=True,           # Sastrawi stemming
    enable_slang_normalization=False,# Slang dict (optional)
)
```

No source code modifications are needed to change the pipeline behavior.
