import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target  # Median house price in $100k

print(X.describe())
print(f"\nTarget range: ${y.min():.2f}k – ${y.max():.2f}k")
print(f"Features: {housing.feature_names}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

y_train_sc = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
y_test_sc  = scaler_y.transform(y_test.reshape(-1,1)).ravel()


from sklearn.svm import SVR

svr = SVR(
    kernel='rbf',  # K(xi,xj) = exp(-γ||xi-xj||²)
    C=1.0,         # Regularization: penalty for violations
    epsilon=0.1,   # Half-width of the ε-tube
    gamma='scale'  # γ = 1/(n_features × X.var())
)

svr.fit(X_train_sc, y_train_sc)

y_pred_sc = svr.predict(X_test_sc)
y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1,1)).ravel()

print(f"Support vectors: {svr.n_support_[0]}")

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

param_grid = {
    'C':       [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma':   ['scale', 'auto', 0.01, 0.1]
}

svr_cv = GridSearchCV(
    SVR(kernel='rbf'),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
svr_cv.fit(X_train_sc, y_train_sc)

print(f"Best params: {svr_cv.best_params_}")

# Evaluate best model
best = svr_cv.best_estimator_
y_pred = scaler_y.inverse_transform(
    best.predict(X_test_sc).reshape(-1,1)
).ravel()

print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Actual vs Predicted scatter
axes[0].scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price ($100k)')
axes[0].set_ylabel('Predicted Price ($100k)')
axes[0].set_title('SVR: Actual vs Predicted')

# 2. Residuals plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price ($100k)')
axes[1].set_ylabel('Residual')
axes[1].set_title('Residuals vs Fitted')

plt.tight_layout()
plt.savefig('svr_results.png', dpi=150, bbox_inches='tight')
plt.show()