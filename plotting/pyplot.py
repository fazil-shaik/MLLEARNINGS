import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
# Generate data
X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=5,
    min_samples_leaf=4,
    min_samples_split=10,
    random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_error = model.predict(X_train)
test_error = model.predict(X_test)

print("mse of train data: ",mean_squared_error(y_train, train_error))
print("mse of test data: ",mean_squared_error(y_test, test_error))
print("r2 score of train data: ",r2_score(y_train, train_error))
print("r2 score of test data: ",r2_score(y_test, test_error))   


# # # Plot
plt.scatter(y_test, y_pred,color='orange',label="Predicted vs Actual",linewidths=2)
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)])
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()


#feature wise predictin
X_sorted = np.sort(X,axis=1)
y_pred_full = model.predict(X_sorted)

plt.scatter(X, y)
plt.plot(X_sorted, y_pred_full,color="#28992c")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Random Forest Regression Curve")
plt.show()


residuals = y_test - y_pred

plt.scatter(y_pred, residuals,color="#592727")
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

X_new = np.linspace(0, 10, 400).reshape(-1, 1)
y_new_pred = model.predict(X_new)

plt.scatter(X, y,color='green')                     # original data
plt.plot(X_new, y_new_pred, color="#262636")           # prediction line
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Random Forest - New Value Prediction")
plt.show()
