import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Data
X = np.array([
    [600, 2, 1], [700, 2, 1], [800, 3, 1], [900, 2, 1],
    [1000, 3, 1], [1100, 3, 1], [1200, 4, 1], [1300, 3, 1],
    [1400, 4, 1], [1500, 3, 1], [1600, 4, 1], [1700, 4, 1],
    [1800, 5, 1], [1900, 4, 1], [2000, 5, 1], [2100, 5, 1],
    [2200, 6, 1], [2300, 5, 1], [2400, 6, 1], [2500, 6, 1],
    [2600, 7, 1], [2700, 6, 1], [2800, 7, 1], [2900, 7, 1],
    [3000, 8, 1], [3100, 7, 1], [3200, 8, 1], [3300, 8, 1],
    [3400, 9, 1], [3500, 9, 1]
])

y = np.array([
    35, 38, 42, 45, 48, 52, 56, 58, 62, 65,
    69, 72, 76, 78, 82, 85, 88, 92, 95, 98,
    102, 105, 108, 112, 115, 118, 122, 125, 128, 132
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (VERY IMPORTANT)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Prediction
y_pred = lasso.predict(X_test)

# Metrics
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Coefficients
print("Intercept:", lasso.intercept_)
print("Coefficients [Area, Rooms, Constant]:", lasso.coef_)


new_X_value = np.array([[2000, 5, 1]])
new_X_value = scaler.transform(new_X_value)

y_new_prediction = lasso.predict(new_X_value)
print("Predicted house price:", y_new_prediction[0])


plt.figure(figsize=(8,5))

for i in range(len(y_test)):
    plt.scatter(y_test[i], y_pred[i], color='red')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)],linewidth=2)  # perfect line

plt.show()
