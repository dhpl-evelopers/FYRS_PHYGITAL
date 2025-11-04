import joblib
import onnxruntime as ort
import pandas as pd
import numpy as np

# Load the saved models
clf = joblib.load("model/random_forest.pkl")
onnx_session = ort.InferenceSession("model/random_forest.onnx")

# Example input (same feature order as during training)
columns = [
    "Who are you purchasing for?", "Gender", "Relation", "Profession",
    "Occasion", "Purpose", "Typical Day", "Weekend Preference",
    "Work Dress", "Social Dress", "Trip Preference", "Waiting Line",
    "Artwork Response", "Mother Response", "Last Minute Plans"
]

# Dummy input example — replace with actual encoded values or data
sample_data = pd.DataFrame([[1, 0, 3, 2, 1, 0, 1, 2, 1, 2, 0, 1, 3, 0, 2]], columns=columns)

# ✅ Predict using joblib model
pred_sklearn = clf.predict(sample_data)
print("Prediction (joblib):", pred_sklearn)

# ✅ Predict using ONNX model
input_name = onnx_session.get_inputs()[0].name
pred_onnx = onnx_session.run(None, {input_name: sample_data.astype(np.float32).to_numpy()})
print("Prediction (ONNX):", pred_onnx[0])
