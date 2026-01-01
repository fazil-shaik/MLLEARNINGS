import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# -----------------------------
# Dataset (outlier fixed)
# -----------------------------
df = pd.DataFrame({
    'hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'exam_scores': [20, 30, 55, 70, 80, 85, 87, 88]
})

# -----------------------------
# Feature selection
# -----------------------------
X = df[['hours']].values
Y = df['exam_scores'].values

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# -----------------------------
# Model training
# -----------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

# -----------------------------
# Predictions
# -----------------------------
y_test_pred = linear_model.predict(X_test)
y_full_pred = linear_model.predict(X)

# -----------------------------
# Evaluation
# -----------------------------
r2_linear = r2_score(Y_test, y_test_pred)

# -----------------------------
# New value prediction
# -----------------------------
new_value = 4.5
new_value_predict = linear_model.predict([[new_value]])

print("\nLinear Regression Model")
print(f"Equation: y = {linear_model.intercept_:.2f} + {linear_model.coef_[0]:.2f}x")
print(f"R² Score: {r2_linear:.4f}")
print(f"Predicted exam score for {new_value} hours: {new_value_predict[0]:.2f}")

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10, 6))

# Actual data points
plt.scatter(X, Y, color='green', s=100, label='Actual data')

# Regression line
plt.plot(X, y_full_pred, color='blue', linewidth=2, label='Regression line')

# Test data points
plt.scatter(X_test, Y_test, color='red', s=120, label='Test data')

plt.xlabel('Hours studied')
plt.ylabel('Exam score')
plt.title('Linear Regression: Hours Studied vs Exam Score')
plt.legend()
plt.grid(True)
plt.show()
