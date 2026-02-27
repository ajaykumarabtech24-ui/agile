# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Create data
X = np.array([1, 2, 3, 4,5]).reshape(-1, 1)   # Independent variable
y = np.array([2, 4, 6, 8,10])                 # Dependent variable

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Get slope and intercept
print("Intercept (a):", model.intercept_)
print("Slope (b):", model.coef_[0])

# Make prediction
new_value = np.array([[5]])
prediction = model.predict(new_value)

print("Prediction for x = 5:", prediction[0])
