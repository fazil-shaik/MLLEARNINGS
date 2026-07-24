import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 300

# Generate data with real relationships between features and price
Data = {
    'Distance': np.random.uniform(2, 15, n_samples),
    'Age':      np.random.randint(2, 50, n_samples),
    'Area':     np.random.randint(500, 1500, n_samples)
}
dataframe = pd.DataFrame(Data)

# Price formula: higher area = more expensive, farther/older = cheaper
noise = np.random.normal(0, 5000, n_samples)
dataframe['Price'] = (
    60    * dataframe['Area']
    - 1500 * dataframe['Distance']
    - 800  * dataframe['Age']
    + 20000
    + noise
)

X = dataframe[['Distance', 'Age', 'Area']]
y = dataframe['Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
residuals = y_test - y_pred

print(f"R² Score     : {r2:.4f}")
print(f"Intercept    : {model.intercept_:.2f}")
print(f"Coefficients :")
for feat, coef in zip(X.columns, model.coef_):
    print(f"  {feat:<12}: {coef:.2f}")

features = X.columns.tolist()
colors   = ['#4C72B0', '#DD8452', '#55A868']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Linear Regression — Full Diagnostic Dashboard',
             fontsize=15, fontweight='bold', y=1.01)

# Row 1: Each feature vs Price
for i, (feat, col) in enumerate(zip(features, colors)):
    ax = axes[0, i]
    ax.scatter(X_test[feat], y_test, color='gray', alpha=0.45, s=30, label='Actual')
    ax.scatter(X_test[feat], y_pred, color=col,    alpha=0.60, s=30, label='Predicted')

    m, b = np.polyfit(X_test[feat], y_pred, 1)
    x_line = np.linspace(X_test[feat].min(), X_test[feat].max(), 100)
    ax.plot(x_line, m * x_line + b, color=col, linewidth=2)

    ax.set_title(f'{feat} vs Price', fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Price')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

# Row 2, Col 1: Actual vs Predicted
ax = axes[1, 0]
ax.scatter(y_test, y_pred, color='#4C72B0', alpha=0.6, s=40,
           edgecolors='white', linewidth=0.4)
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Fit')
ax.set_title('Actual vs Predicted', fontweight='bold')
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
ax.text(0.05, 0.92, f'R² = {r2:.4f}', transform=ax.transAxes,
        fontsize=11, color='darkred', fontweight='bold')

# Row 2, Col 2: Feature Importance (Coefficients)
ax = axes[1, 1]
coefs = model.coef_
bars = ax.barh(features, coefs, color=colors, edgecolor='white', height=0.5)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Feature Importance (Coefficients)', fontweight='bold')
ax.set_xlabel('Coefficient Value')
for bar, val in zip(bars, coefs):
    ax.text(val + (max(abs(coefs)) * 0.02), bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}', va='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

# Row 2, Col 3: Residuals Distribution
ax = axes[1, 2]
ax.hist(residuals, bins=25, color='#55A868', edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Residual')
ax.set_title('Residuals Distribution', fontweight='bold')
ax.set_xlabel('Residual (Actual − Predicted)')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()