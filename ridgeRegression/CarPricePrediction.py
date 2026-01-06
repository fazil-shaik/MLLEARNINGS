import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import r2_score

X_car = [
    [800, 10],
    [1000, 8],
    [1200, 6],
    [1500, 4],
    [1800, 2]
]

y_car = [2.5, 3.2, 4.5, 6.8, 9.0]  # Price in lakhs


LinearModel = LinearRegression()
LinearModel.fit(X_car,y_car)

y_Line_prediction = LinearModel.predict(X_car)

print(f"\n Linear regression ")
print(f"Price = {LinearModel.intercept_:.0f}+{LinearModel.coef_[0]:.3f} * Model")

RidgeModel = Ridge(alpha=4)
RidgeModel.fit(X_car,y_car)
y_ridge_predict = RidgeModel.predict(X_car)
print(f"\n Ridge regression ")
print(f"Price = {RidgeModel.intercept_:.0f}+{RidgeModel.coef_[0]:.3f} * Model")

# plt.figure(figsize=(10,6))
# plt.scatter(X_car[:, 0],y_car,label='Actual Values')
# plt.plot(y_car,y_Line_prediction,label='Linear prediction')
# plt.legend()
# plt.show()

# Convert to numpy arrays
X_car = np.array(X_car)
y_car = np.array(y_car)

plt.figure(figsize=(10,6))

# Actual prices
plt.scatter(X_car[:, 0], y_car, label='Actual Values')

# Linear regression predictions
plt.plot(X_car[:, 0], y_Line_prediction, label='Linear Prediction')

# Ridge regression predictions
plt.plot(X_car[:, 0], y_ridge_predict, label='Ridge Prediction')

plt.xlabel("Engine Capacity (cc)")
plt.ylabel("Car Price (Lakhs)")
plt.legend()
plt.show()

r2_linear = r2_score(y_car, y_Line_prediction)
r2_ridge = r2_score(y_car, y_ridge_predict)

print(f"Linear Regression R² score: {r2_linear:.4f}")
print(f"Ridge Regression R² score:  {r2_ridge:.4f}")
