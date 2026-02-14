import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_friedman1(n_samples=2000, noise=1.0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_weak = GradientBoostingRegressor(
    n_estimators=10,      # very few trees
    learning_rate=0.1,
    max_depth=2,
    random_state=42
)

model_weak.fit(X_train, y_train)

pred_train_w = model_weak.predict(X_train)
pred_test_w = model_weak.predict(X_test)

print("WEAK MODEL")
print("Train R2:", r2_score(y_train, pred_train_w))
print("Test R2:", r2_score(y_test, pred_test_w))


model_strong = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,       # stochastic boosting
    random_state=42
)

model_strong.fit(X_train, y_train)

pred_train_s = model_strong.predict(X_train)
pred_test_s = model_strong.predict(X_test)

print("\nSTRONG MODEL")
print("Train R2:", r2_score(y_train, pred_train_s))
print("Test R2:", r2_score(y_test, pred_test_s))

train_errors = []
test_errors = []

for y_train_pred, y_test_pred in zip(
    model_strong.staged_predict(X_train),
    model_strong.staged_predict(X_test)
):
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

plt.figure(figsize=(8,5))
plt.plot(train_errors, label="Train Error")
plt.plot(test_errors, label="Test Error")
plt.xlabel("Number of Trees")
plt.ylabel("MSE")
plt.legend()
plt.title("Gradient Boosting Learning Curve")
plt.show()

importances = model_strong.feature_importances_

plt.bar(range(len(importances)), importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()


from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [200, 300],
    "learning_rate": [0.03, 0.05],
    "max_depth": [2, 3],
    "subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
