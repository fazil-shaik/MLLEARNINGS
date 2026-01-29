from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

# Model
model = LogisticRegression()
model.fit(X, y)

# Predictions for training data
y_predict = model.predict(X)

# New value
new_value = np.array([[1.8]])
y_prob_predict = model.predict_proba(new_value)[:, 1]
print(f"Logistic regression probability is {y_prob_predict[0]:.3f}")

# Smooth curve for sigmoid
X_test = np.linspace(1, 4, 100).reshape(-1, 1)
y_prob_curve = model.predict_proba(X_test)[:, 1]

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='red', label='Actual data')
plt.plot(X_test, y_prob_curve, color='blue', label='Sigmoid curve')
plt.scatter(new_value, y_prob_predict, color='purple', marker='*', s=200,
            label='Prediction for x=1.8')

plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.show()
