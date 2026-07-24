import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.rand(100,2)*10
y = 4*X+3 + np.random.randn(100,2)*5

w=0
b=0
learning_rate = 0.1
epochs = 100
n = len(X)


for i in range(epochs):
    
    # Predictions
    y_pred = w * X + b
    
    # Compute gradients
    dw = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Print loss every 10 iterations
    if i % 10 == 0:
        loss = np.mean((y - y_pred) ** 2)
        print(f"Epoch {i}, Loss: {loss:.4f}")


print("Final weight:", w)
print("Final bias:", b)

plt.scatter(X, y)
plt.plot(X, w*X + b, color='red')
plt.title("Gradient Descent Linear Regression")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate nonlinear dataset
X, y = make_friedman1(n_samples=2000, noise=1.0, random_state=42)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model with Early Stopping
model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    validation_fraction=0.1,
    n_iter_no_change=10,   # early stopping
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train R2:", r2_score(y_train, y_train_pred))
print("Test R2:", r2_score(y_test, y_test_pred))
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))



train_errors = []
test_errors = []

for y_train_stage, y_test_stage in zip(
    model.staged_predict(X_train),
    model.staged_predict(X_test)
):
    train_errors.append(mean_squared_error(y_train, y_train_stage))
    test_errors.append(mean_squared_error(y_test, y_test_stage))

plt.figure(figsize=(8,5))
plt.plot(train_errors, label="Train Error")
plt.plot(test_errors, label="Test Error")
plt.xlabel("Number of Trees")
plt.ylabel("MSE")
plt.title("Gradient Boosting Learning Curve")
plt.legend()
plt.show()



plt.figure(figsize=(8,5))
plt.bar(range(X.shape[1]), model.feature_importances_)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()