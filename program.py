import numpy as np
from sklearn.linear_model import LinearRegression

# Data
hours = np.array([[1], [2], [3], [4], [5]])
marks = np.array([40, 50, 60, 65, 80])

# Train model
model = LinearRegression()
model.fit(hours, marks)

# Predict
h = 6  # study hours
pred = model.predict([[h]])

print(f"Predicted marks for {h} hours = {pred[0]:.2f}")
