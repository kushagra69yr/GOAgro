import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("crop_data.csv", sep=None, engine='python')

print(data.head())

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

sample = np.array([[90, 40, 40, 25, 80, 6.5, 200]])
prediction = model.predict(sample)

print("Recommended Crop:", prediction[0])
import joblib

# Save model
joblib.dump(model, "crop_model.pkl")

print("Model saved successfully!")