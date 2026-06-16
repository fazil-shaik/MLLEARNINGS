import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("Loading real mining data...")
df = pd.read_csv('mining_data_real.csv')

# Feature engineering
feature_columns = [
    'price',
    'network_difficulty',
    'hashrate_ths',
    'volume',
    'power_consumption_w',
    'electricity_cost_per_kwh',
    'pool_fee',
    'hardware_age_months'
]

# One-hot encode hardware
hardware_dummies = pd.get_dummies(df['hardware'], prefix='hardware')
X = pd.concat([df[feature_columns], hardware_dummies], axis=1)

# Target: monthly profit
y = df['monthly_profit_usd']

# Remove any rows with missing values
X = X.dropna()
y = y[X.index]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test: {X_test.shape[0]} samples")

# Train models
models = {}
results = []

linear = LinearRegression()
linear.fit(X_train, y_train)
models['linear'] = linear
results.append({
    'Model': 'Linear',
    'R²': linear.score(X_test, y_test),
    'RMSE': np.sqrt(mean_squared_error(y_test, linear.predict(X_test)))
})

lasso = Lasso(alpha=0.01, max_iter=5000)
lasso.fit(X_train, y_train)
models['lasso'] = lasso
results.append({
    'Model': 'Lasso',
    'R²': lasso.score(X_test, y_test),
    'RMSE': np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))
})

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
models['ridge'] = ridge
results.append({
    'Model': 'Ridge',
    'R²': ridge.score(X_test, y_test),
    'RMSE': np.sqrt(mean_squared_error(y_test, ridge.predict(X_test)))
})

poly = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])
poly.fit(X_train, y_train)
models['polynomial'] = poly
results.append({
    'Model': 'Polynomial',
    'R²': poly.score(X_test, y_test),
    'RMSE': np.sqrt(mean_squared_error(y_test, poly.predict(X_test)))
})

import os
os.makedirs('models', exist_ok=True)

for name, model in models.items():
    with open(f'models/mining_{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved mining_{name}.pkl")

with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
results_df = pd.DataFrame(results)
print(results_df.round(4))

lasso_coef = dict(zip(X.columns, lasso.coef_))
selected = {k: v for k, v in lasso_coef.items() if v != 0}
print(f"\n Lasso selected {len(selected)} features:")
for k, v in sorted(selected.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"   {k}: {v:.4f}")