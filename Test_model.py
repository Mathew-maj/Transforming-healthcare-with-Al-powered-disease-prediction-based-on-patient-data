import joblib
import numpy as np

model = joblib.load('model.pkl')
sample = np.array([[1, 0, 1, 0, 1, 0, 0]])  # Replace with actual features
prediction = model.predict(sample)

print("Predicted disease:", prediction[0])
