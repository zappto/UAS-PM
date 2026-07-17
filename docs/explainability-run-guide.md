# Explainability Run Guide

This document provides instructions on how to run the Explainability stage of the Machine Learning project.

## 1. Required Python Packages

The following packages are strictly required to run the explainability module:
- `pandas`
- `numpy`
- `scikit-learn`
- `shap`
- `matplotlib`
- `joblib`
- `scipy`
- `jupyter`

## 2. Installation Commands

If you haven't installed the dependencies, you can do so by running the following command from the project root:

```bash
pip install -r requirements.txt
```
*(Ensure `shap` and `matplotlib` are included in your requirements.txt)*

Alternatively, install them manually:
```bash
pip install pandas numpy scikit-learn shap matplotlib joblib scipy jupyter
```

## 3. Expected Project Structure

Ensure your project structure looks like this before proceeding:

```text
project/
├── dataset/
│   └── processed/
│       ├── X_test.npz
│       └── y_test.csv
├── docs/
│   └── explainability-run-guide.md
├── models/
│   ├── logistic_regression_best.joblib
│   ├── naive_bayes_best.joblib
│   ├── svm_best.joblib
│   └── tfidf_vectorizer.joblib
├── notebooks/
│   └── 08_explainability.ipynb
└── src/
    ├── config/
    │   ├── settings.py
    │   └── explainability_config.py
    ├── explainability/
    │   ├── explainer_factory.py
    │   ├── shap_explainer.py
    │   ├── fallback_explainer.py
    │   ├── local_explanation.py
    │   ├── global_explanation.py
    │   ├── text_highlighter.py
    │   ├── report_generator.py
    │   ├── validator.py
    │   └── persistence.py
    └── visualization/
        └── explainability_viz.py
```

## 4. Required Input Files

You must have successfully run the previous training and tuning stages so that the following files exist:

**Models:**
- `models/logistic_regression_best.joblib`
- `models/naive_bayes_best.joblib`
- `models/svm_best.joblib`
- `models/tfidf_vectorizer.joblib`

**Data:**
- `dataset/processed/X_test.npz`
- `dataset/processed/y_test.csv`

## 5. How to Execute the Notebook

The primary way to generate explainability artifacts is via the Jupyter Notebook.

1. Start Jupyter from your terminal:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/08_explainability.ipynb`.
3. Run all cells sequentially.
4. The notebook will automatically generate visualizations, CSVs, and the final Markdown report.

## 6. How to Execute Python Modules

The modules in `src/explainability` are designed to be reusable in the Streamlit application. They are not intended to be run directly as scripts via `python src/explainability/xxx.py`. Instead, import them into your deployment code as demonstrated in the notebook.

## 7. Expected Generated Files

After successfully running the notebook, the following artifacts will be created in `reports/explainability/`:

```text
reports/explainability/
├── explainability_summary.md
├── global_feature_importance.csv
├── class_feature_importance.json
├── local_explanations.json
├── highlighted_words.json
├── figures/
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── feature_importance.png
│   ├── class_importance.png
│   └── local_explanation.png
└── raw/
    └── shap_values.npy
```

## 8. Troubleshooting

- **Missing Files:** Ensure `models/` and `dataset/processed/` contain the expected outputs from previous phases. Rerun the tuning and preprocessing notebooks if necessary.
- **Memory Issues:** SHAP values can be memory-intensive. If your kernel crashes, try passing a smaller background dataset (`x_train`) or a subset of `X_test` to the explainer.
- **Matplotlib Backend Issues:** If visualizations are not rendering or causing errors, ensure you are running the notebook in an environment that supports GUI outputs, or verify that the `ExplainabilityVisualizer` closes plots properly after saving.

## 9. Common Errors

| Error | Possible Cause | Solution |
|-------|----------------|----------|
| `FileNotFoundError` | Missing models or vectorizer | Run the training and tuning phase first. |
| `ValidationError: Model missing 'predict'` | Incorrect object loaded | Ensure you saved the scikit-learn model, not a raw dict. |
| `IndexError: out of bounds for dimension` | Features mismatch | Ensure you use the exact TF-IDF vectorizer that was fit on the training data. |

## 10. Validation Checklist

- [ ] All models load successfully.
- [ ] TF-IDF vectorizer loads and feature names match the model dimension.
- [ ] Factory automatically selects `ShapExplainer` for LR and SVM.
- [ ] Factory selects `FallbackExplainer` for Naive Bayes.
- [ ] Global feature importance CSV is generated.
- [ ] Local explanation JSON is generated.
- [ ] All 5 Matplotlib figures are saved in `reports/explainability/figures/`.
- [ ] `explainability_summary.md` is fully populated.
