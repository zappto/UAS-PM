# Error Analysis Run Guide

## Overview

This guide explains how to execute the Error Analysis pipeline for the **Cyberbullying Type Classification** research project.

This stage analyzes *why* models make mistakes. It computes error percentages, identifies the most difficult classes (high recall failure), finds frequently confused class pairs, and performs basic text pattern analysis on misclassified samples. 

It does **not** evaluate model metrics or implement explainability (SHAP).

---

## 1. Required Python Packages

| Package | Purpose | Version |
|---------|---------|---------|
| pandas | Data manipulation | 2.0+ |
| matplotlib | Generating error visualizations | 3.7+ |
| joblib | Loading models (if predictions missing) | 1.3+ |
| scikit-learn | Loading data features (if needed) | 1.3+ |
| scipy | Handling sparse matrices | 1.10+ |

---

## 2. Installation Commands

```bash
pip install pandas matplotlib joblib scikit-learn scipy
```

---

## 3. Expected Project Structure

```
project/
├── dataset/processed/
│   ├── final_dataset_clean.csv  ← INPUT (Original Text)
│   ├── X_test.npz               ← INPUT (If generating predictions)
│   └── y_test.csv               ← INPUT (True Labels)
│
├── reports/evaluation/predictions/
│   └── *_predictions.csv        ← INPUT (or automatically generated here)
│
├── src/
│   ├── config/
│   │   └── error_analysis_config.py
│   └── error_analysis/
│       ├── validator.py
│       ├── error_loader.py
│       ├── misclassification.py
│       ├── class_analysis.py
│       ├── confusion_analysis.py
│       ├── text_pattern_analysis.py
│       ├── error_statistics.py
│       ├── report_generator.py
│       └── persistence.py
│
├── notebooks/
│   └── 07_error_analysis.ipynb
│
└── reports/error_analysis/      ← OUTPUTS DIRECTORY
    ├── error_summary.md
    ├── error_statistics.csv
    ├── difficult_classes.csv
    ├── confusion_analysis.csv
    ├── frequent_error_words.csv
    ├── figures/
    │   ├── error_distribution.png
    │   └── top_confused_classes.png
    └── raw/
        └── *_misclassified.csv
```

---

## 4. Required Input Files

The pipeline will first look for prediction files in `reports/evaluation/predictions/`. 
If they do not exist, it will **automatically** use `models/*_best.joblib` and `dataset/processed/X_test.npz` to generate them.

---

## 5. How to Execute — Jupyter Notebook

```bash
cd /home/zapp/Kampus/PM
jupyter notebook notebooks/07_error_analysis.ipynb
```

Run each cell sequentially.

---

## 6. How to Execute — Python Script

```bash
cd /home/zapp/Kampus/PM

python3 -c "
import sys; sys.path.insert(0, '.')
from src.error_analysis.error_loader import load_and_prepare_errors
from src.error_analysis.misclassification import analyze_misclassifications
from src.error_analysis.class_analysis import analyze_class_errors, identify_difficult_classes
from src.error_analysis.confusion_analysis import analyze_confusions, aggregate_top_confusions
from src.error_analysis.text_pattern_analysis import analyze_text_characteristics, find_frequent_error_words
from src.error_analysis.error_statistics import generate_error_statistics
from src.error_analysis.report_generator import generate_error_summary
from src.error_analysis.persistence import save_error_analysis_artifacts
from src.config.error_analysis_config import ErrorAnalysisConfig

# 1. Load Data
df = load_and_prepare_errors()

# 2. Analyze
misclass = analyze_misclassifications(df)
stats_df = generate_error_statistics(misclass)

class_dfs = {m: analyze_class_errors(df, f'pred_{m}') for m in stats_df.index}
diff_classes = identify_difficult_classes(class_dfs)

conf_dfs = {m: analyze_confusions(df, f'pred_{m}') for m in stats_df.index}
top_confs = aggregate_top_confusions(conf_dfs)

best_model = stats_df['error_percentage'].idxmin()
freq_words = find_frequent_error_words(misclass[best_model]['misclassified_df'])
raw_dfs = {m: d['misclassified_df'] for m, d in misclass.items()}

# 3. Save
summary = generate_error_summary(stats_df, diff_classes, top_confs)
save_error_analysis_artifacts(stats_df, diff_classes, top_confs, freq_words, raw_dfs, summary, ErrorAnalysisConfig())

print('\n✓ Error Analysis complete.')
"
```

---

## 7. Expected Generated Files

| File | Description |
|------|-------------|
| `error_summary.md` | Main summary of the error analysis |
| `error_statistics.csv` | Error vs Correct counts and percentages |
| `difficult_classes.csv` | Classes ranked by false negative rate |
| `confusion_analysis.csv` | Top confused pairs aggregated |
| `frequent_error_words.csv`| Simple token frequency of errors (Best model) |
| `figures/*.png` | Visualizations of errors and confusions |
| `raw/*_misclassified.csv` | The actual text samples that were incorrectly predicted |

---

## 8. Troubleshooting

### FileNotFoundError: Clean dataset missing

The error loader tries to load `final_dataset_clean.csv` to analyze the actual text that failed. Ensure Notebook 02 (Preprocessing) completed successfully.

### "Cannot perfectly map test subset to original text" Warning

If the `y_test.csv` file was saved without original dataset indices, the script will map `<text_unavailable>` to prevent crashes. The pattern analysis will be skipped. Ensure `y_test.csv` was generated properly during Feature Engineering.

---

## 9. Validation Checklist

After running, verify:

```bash
# 1. Check reports
ls -la reports/error_analysis/

# 2. Check figures
ls -la reports/error_analysis/figures/

# 3. Read summary
cat reports/error_analysis/error_summary.md
```
