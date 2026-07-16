import json
import traceback
from src.explainability.inference import InferencePipeline

try:
    print("Initializing inference pipeline...")
    pipeline = InferencePipeline()
    print("Running prediction...")
    text = "Kamu kontol sap"
    result = pipeline.predict_with_explanation(text)
    
    print("\nResult:")
    print(json.dumps(result, indent=2))
except Exception as e:
    traceback.print_exc()
