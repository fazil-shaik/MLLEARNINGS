import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load your synthetic data
print("Loading synthetic coffee data...")
df = pd.read_csv('./data/coffee_synthetic.csv')  # Replace with your actual file path

# Prepare features and targets
feature_columns = ['roast_time_min', 'temp_ramp_c_min', 'moisture_pct', 
                   'density_g_ml', 'airflow']
target_columns = ['acidity', 'sweetness', 'body']

X = df[feature_columns]
y_acidity = df['acidity']
y_sweetness = df['sweetness']
y_body = df['body']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_acidity, test_size=0.2, random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}\n")

# Dictionary to store all models
models = {}

# 1. Linear Regression
print("Training Linear Regression...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
models['linear'] = linear_model

# Evaluate
y_pred = linear_model.predict(X_test)
print(f"Linear Regression - R²: {r2_score(y_test, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# 2. Lasso Regression
print("\nTraining Lasso Regression...")
lasso_model = Lasso(alpha=0.1, random_state=42)
lasso_model.fit(X_train, y_train)
models['lasso'] = lasso_model

# Evaluate
y_pred = lasso_model.predict(X_test)
print(f"Lasso Regression - R²: {r2_score(y_test, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Feature coefficients: {dict(zip(feature_columns, lasso_model.coef_))}")

# 3. Ridge Regression
print("\nTraining Ridge Regression...")
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train, y_train)
models['ridge'] = ridge_model

# Evaluate
y_pred = ridge_model.predict(X_test)
print(f"Ridge Regression - R²: {r2_score(y_test, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# 4. Polynomial Regression (degree 2)
print("\nTraining Polynomial Regression...")
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])
poly_model.fit(X_train, y_train)
models['polynomial'] = poly_model

# Evaluate
y_pred = poly_model.predict(X_test)
print(f"Polynomial Regression - R²: {r2_score(y_test, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Save all models
print("\nSaving models...")
with open('models/linear.pkl', 'wb') as f:
    pickle.dump(linear_model, f)

with open('models/lasso.pkl', 'wb') as f:
    pickle.dump(lasso_model, f)

with open('models/ridge.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)

with open('models/polynomial.pkl', 'wb') as f:
    pickle.dump(poly_model, f)

print("All models saved successfully in 'models/' directory!")

# Save feature names for reference
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("\nTraining completed!")
print("\nModel Performance Summary:")
print("-" * 50)
for name in ['linear', 'lasso', 'ridge', 'polynomial']:
    with open(f'models/{name}.pkl', 'rb') as f:
        model = pickle.load(f)
    if name == 'polynomial':
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    print(f"{name.capitalize():12} R²: {r2_score(y_test, y_pred):.4f}")