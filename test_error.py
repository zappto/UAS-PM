import sys
import traceback
import joblib
import scipy.sparse
import numpy as np

try:
    X_test = scipy.sparse.load_npz("dataset/processed/X_test.npz")
    model = joblib.load("models/logistic_regression_best.joblib")

    import shap
    
    # Use zero masker with 2D shape (1, num_features)
    masker = shap.maskers.Independent(np.zeros((1, model.coef_.shape[1])))
    explainer = shap.LinearExplainer(model, masker)
    
    # Get shap values
    subset = X_test[:5]
    shap_vals = explainer.shap_values(subset)
    print(f"shap_vals type: {type(shap_vals)}")
    if isinstance(shap_vals, list):
        print(f"shap_vals len: {len(shap_vals)}")
        print(f"shap_vals[0] shape: {shap_vals[0].shape}")
    else:
        print(f"shap_vals shape: {shap_vals.shape}")

except Exception as e:
    traceback.print_exc()
