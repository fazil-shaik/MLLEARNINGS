import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load REAL dataset

data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())


# 2. Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Create Random Forest

model = RandomForestRegressor(
    n_estimators=100,     # 100 decision trees
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# 4. Train

model.fit(
    X_train,
    y_train
)


# 5. Predict on unseen data

y_pred = model.predict(
    X_test
)


# 6. Evaluate

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n-------------------------")
print("MODEL PERFORMANCE")
print("-------------------------")

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)


# 7. Feature importance

print("\n-------------------------")
print("FEATURE IMPORTANCE")
print("-------------------------")

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(
        f"{feature:15} {importance:.3f}"
    )


# 8. Visualize actual vs predicted

plt.scatter(
    y_test,
    y_pred,
    alpha=0.3
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")

plt.title(
    "Random Forest Regression"
)

# Perfect prediction line

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.tight_layout()
plt.show()


# 9. REAL prediction for a new house

new_house = X.iloc[[0]].copy()

prediction = model.predict(
    new_house
)

print("\n-------------------------")
print("NEW HOUSE PREDICTION")
print("-------------------------")

print(
    "Predicted value:",
    prediction[0] * 100000
)

print("dollars")