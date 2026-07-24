import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
n_samples = 200
X = np.random.uniform(-3, 3, (n_samples, 1))
y = 2.5 * X.ravel() + 1.2 * X.ravel()**2 + np.sin(3 * X.ravel()) + np.random.normal(0, 0.5, n_samples)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial features for non-linear models
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Standardize for regularization
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 2. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_poly_scaled)

# 3. Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_poly_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_poly_scaled)

# 4. Elastic Net
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(X_train_poly_scaled, y_train)
y_pred_en = en.predict(X_test_poly_scaled)

# Print results
print("="*50)
print(f"{'Model':<15} {'R² Score':<12} {'RMSE':<12}")
print("="*50)
print(f"{'Linear Reg':<15} {r2_score(y_test, y_pred_lr):<12.4f} {np.sqrt(mean_squared_error(y_test, y_pred_lr)):<12.4f}")
print(f"{'Ridge':<15} {r2_score(y_test, y_pred_ridge):<12.4f} {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):<12.4f}")
print(f"{'Lasso':<15} {r2_score(y_test, y_pred_lasso):<12.4f} {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):<12.4f}")
print(f"{'ElasticNet':<15} {r2_score(y_test, y_pred_en):<12.4f} {np.sqrt(mean_squared_error(y_test, y_pred_en)):<12.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
predictions = [y_pred_lr, y_pred_ridge, y_pred_lasso, y_pred_en]
titles = ['Linear Regression', 'Ridge (L2)', 'Lasso (L1)', 'ElasticNet']

for i, (ax, title, pred) in enumerate(zip(axes.flat, titles, predictions)):
    ax.scatter(X_test.ravel(), y_test, alpha=0.6, label='Actual')
    ax.scatter(X_test.ravel(), pred, alpha=0.6, c='red', label='Predicted')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()