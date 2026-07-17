# Streamlit Run Guide

This document provides instructions on how to run the Streamlit deployment of the Machine Learning project.

## 1. Required Python Packages

The following packages are strictly required to run the Streamlit application:
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `shap`
- `joblib`
- `scipy`

## 2. Installation Commands

If you haven't installed the dependencies, run the following command from the project root:

```bash
pip install -r requirements.txt
```

Alternatively, install them manually:
```bash
pip install streamlit pandas numpy matplotlib scikit-learn shap joblib scipy
```

## 3. Required Project Structure

Ensure your project structure looks like this before proceeding:

```text
project/
├── models/
│   ├── logistic_regression_best.joblib
│   └── tfidf_vectorizer.joblib
├── reports/
│   └── evaluation/
│       └── comparison/
│           └── model_comparison.csv
├── src/
│   └── explainability/
│       └── inference.py
├── streamlit/
│   ├── app.py
│   ├── pages/
│   ├── components/
│   └── utils/
└── docs/
    └── streamlit-run-guide.md
```

## 4. Required Model Files

The Inference pipeline defaults to loading:
- `models/logistic_regression_best.joblib`
- `models/tfidf_vectorizer.joblib`

You must have successfully run the training, tuning, and explainability phases so these files exist.

## 5. Required Reports

- `reports/evaluation/comparison/model_comparison.csv`: Loaded by the *Model Comparison* page.
- `reports/evaluation/model_metadata.json`: Loaded by the *Model Information* page (if it exists, otherwise falls back to defaults).

## 6. Running Streamlit

To launch the application, run the following command from the project root (`project/`):

```bash
streamlit run streamlit/app.py
```

Streamlit will automatically open a browser window at `http://localhost:8501`.

## 7. Troubleshooting

- **ModuleNotFoundError**: Streamlit cannot find `src` or `utils`. Ensure you are running the `streamlit run` command from the **root directory of the project**, not from inside the `streamlit/` folder. The application automatically appends the project root to the `sys.path`.
- **FileNotFoundError**: Missing models or TF-IDF. Rerun the ML pipeline (notebooks 01 to 07).
- **Streamlit caches not clearing**: If you manually updated a model and the UI doesn't reflect it, go to the top right menu in Streamlit and click "Clear Cache", or press `C`.
- **Memory Issues**: The app caches the model in memory. If you face RAM constraints, close unused applications or restart the Streamlit server.

## 8. Common Errors

| Error | Possible Cause | Solution |
|-------|----------------|----------|
| `FileNotFoundError` on `model_comparison.csv` | The evaluation report was not generated. | The app will display a warning inside the Model Comparison page instead of crashing. Generate the reports to see the charts. |
| `IndexError` inside SHAP | Incompatible model/masker | Ensure `src.explainability.shap_explainer` uses the 2D array fix (`np.zeros((1, num_features))`). |

## 9. Validation Checklist

- [ ] Run `streamlit run streamlit/app.py` from root.
- [ ] Verify the *Home* page loads without errors.
- [ ] Navigate to *Prediction*, click an Example button, and click *Predict*.
- [ ] Verify Prediction Card, Probability Chart, and Highlighted Text render correctly.
- [ ] Verify Prediction History updates.
- [ ] Verify Download JSON button works.
- [ ] Check *Model Comparison* page renders the table and chart.
- [ ] Toggle Dark Mode in Streamlit settings and ensure text highlights remain readable.
