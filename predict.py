import numpy as np
import joblib

# Load model
model = joblib.load("crop_model.pkl")

# Example input
sample = np.array([[90, 40, 40, 25, 80, 6.5, 200]])

prediction = model.predict(sample)

print("Recommended Crop:", prediction[0])