from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
dataset = pd.read_csv('insurance.csv')

print(dataset.head())

result = dataset.isnull().sum()
print(result)

sns.scatterplot(
    x=dataset['age'],
    y=dataset['charges'],
    hue=dataset['bmi']
)
plt.show()

X = dataset.drop(columns=['charges', 'region'])
y = dataset['charges']

X['sex'] = X['sex'].map({
    'female': 1,
    'male': 0
})
X['smoker'] = X['smoker'].map({
    'yes': 1,
    'no': 0
})

print(X.head())

#train testing splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model selection 
LinearModel = LinearRegression()
LinearModel.fit(X_train, y_train)

#prediction
y_pred = LinearModel.predict(X_test)

#model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
# Better regression metrics instead of accuracy
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)



fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(x=y_test, y=y_pred, ax=ax[0])
ax[0].set_title('Actual vs Predicted')
ax[0].set_xlabel('Actual Charges')
ax[0].set_ylabel('Predicted Charges')
sns.histplot(y_test - y_pred, ax=ax[1], kde=True)
ax[1].set_title('Residuals Distribution')
ax[1].set_xlabel('Residuals')
plt.tight_layout()
plt.show()





# Load dataset
dataset = pd.read_csv('insurance.csv')

# Prepare features and target
X = dataset.drop(columns=['charges', 'region'])
y = dataset['charges']

# Encode categorical variables
X['sex'] = X['sex'].map({'female': 1, 'male': 0})
X['smoker'] = X['smoker'].map({'yes': 1, 'no': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# METHOD 1: Manual Polynomial Features
print("="*60)
print("METHOD 1: Manual Polynomial Features")
print("="*60)

# Create polynomial features (degree 2 includes: age, age², age×bmi, etc.)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Polynomial features (degree 2): {X_train_poly.shape[1]}")
print(f"Feature names: {poly.get_feature_names_out(X.columns)[:10]}...")  # First 10

# Train model on polynomial features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predictions
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluation
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"\nPolynomial Regression (degree 2) Performance:")
print(f"RMSE: ${rmse_poly:.2f}")
print(f"R² Score: {r2_poly:.4f}")

# METHOD 2: Pipeline (Easier & Cleaner)
print("\n" + "="*60)
print("METHOD 2: Pipeline Approach")
print("="*60)

# Create pipeline with polynomial features + linear regression
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear_reg', LinearRegression())
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Predictions
y_pred_pipeline = pipeline.predict(X_test)

# Evaluation
rmse_pipeline = np.sqrt(mean_squared_error(y_test, y_pred_pipeline))
r2_pipeline = r2_score(y_test, y_pred_pipeline)

print(f"Pipeline Performance:")
print(f"RMSE: ${rmse_pipeline:.2f}")
print(f"R² Score: {r2_pipeline:.4f}")

# COMPARE DIFFERENT DEGREES (Find optimal polynomial degree)
print("\n" + "="*60)
print("COMPARING DIFFERENT POLYNOMIAL DEGREES")
print("="*60)

degrees = [1, 2, 3, 4]
train_scores = []
test_scores = []

for degree in degrees:
    # Create pipeline
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    # Train
    pipe.fit(X_train, y_train)
    
    # Calculate R² scores
    train_r2 = pipe.score(X_train, y_train)
    test_r2 = pipe.score(X_test, y_test)
    
    train_scores.append(train_r2)
    test_scores.append(test_r2)
    
    print(f"Degree {degree}: Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")

# VISUALIZATION 1: Compare Linear vs Polynomial for Age (single feature)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature 1: Age vs Charges
age_range = np.linspace(X['age'].min(), X['age'].max(), 100).reshape(-1, 1)

# For visualization, create models using only Age + Smoker
X_age_only = X[['age', 'smoker']].copy()

X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(
    X_age_only, y, test_size=0.2, random_state=42
)

# Linear model (degree 1)
linear_age = LinearRegression()
linear_age.fit(X_train_age, y_train_age)

# Polynomial model (degree 3)
poly_age = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('linear', LinearRegression())
])
poly_age.fit(X_train_age, y_train_age)

# Create prediction data
age_grid = np.linspace(18, 65, 100)
smoker_status = [0, 1]  # 0 = non-smoker, 1 = smoker

for smoker in smoker_status:
    X_grid = pd.DataFrame({
        'age': age_grid,
        'smoker': [smoker] * len(age_grid)
    })
    
    y_pred_linear = linear_age.predict(X_grid)
    y_pred_poly = poly_age.predict(X_grid)
    
    label = "Smoker" if smoker == 1 else "Non-smoker"
    axes[0].plot(age_grid, y_pred_linear, '--', label=f'Linear - {label}')
    axes[1].plot(age_grid, y_pred_poly, label=f'Polynomial (deg 3) - {label}')

# Add scatter points
smoker_colors = {0: 'blue', 1: 'red'}
for smoker in smoker_status:
    subset = X_test_age[X_test_age['smoker'] == smoker]
    y_subset = y_test_age[X_test_age['smoker'] == smoker]
    color = smoker_colors[smoker]
    label = "Smoker" if smoker == 1 else "Non-smoker"
    axes[0].scatter(subset['age'], y_subset, alpha=0.3, color=color, label=f'Data - {label}')
    axes[1].scatter(subset['age'], y_subset, alpha=0.3, color=color)

axes[0].set_title('Linear Regression (Degree 1)')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Charges ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Polynomial Regression (Degree 3)')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Charges ($)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# VISUALIZATION 2: Learning Curves (Degree Comparison)
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, 'bo-', label='Train R²', linewidth=2, markersize=8)
plt.plot(degrees, test_scores, 'ro-', label='Test R²', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Model Complexity vs Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(degrees)

# Highlight best degree
best_degree = degrees[np.argmax(test_scores)]
plt.axvline(x=best_degree, color='green', linestyle='--', alpha=0.5)
plt.text(best_degree + 0.1, min(train_scores), f'Best: Degree {best_degree}', fontsize=10)

plt.tight_layout()
plt.show()

# VISUALIZATION 3: Actual vs Predicted (Best Model)
best_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
    ('linear', LinearRegression())
])
best_pipe.fit(X_train, y_train)
y_pred_best = best_pipe.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_best, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Charges ($)')
axes[0].set_ylabel('Predicted Charges ($)')
axes[0].set_title(f'Actual vs Predicted (Degree {best_degree}, R² = {test_scores[best_degree-1]:.3f})')

# Residuals
residuals = y_test - y_pred_best
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residuals ($)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')

plt.tight_layout()
plt.show()

# FINAL COMPARISON: Linear vs Polynomial
print("\n" + "="*60)
print("FINAL COMPARISON: Linear vs Polynomial Regression")
print("="*60)

# Linear Regression (degree 1)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

print(f"\n{'Metric':<20} {'Linear':<20} {'Polynomial (deg '+str(best_degree)+')':<20}")
print("-"*60)
print(f"{'RMSE':<20} ${rmse_linear:.2f}{'':<10} ${rmse_pipeline:.2f}")
print(f"{'R² Score':<20} {r2_linear:.4f}{'':<12} {r2_pipeline:.4f}")
print(f"{'Improvement':<20} {'-':<20} {(r2_pipeline - r2_linear)/r2_linear*100:.1f}% better R²")

# Additional metrics
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print(f"{'MAE':<20} ${mae_linear:.2f}{'':<10} ${mae_poly:.2f}")

# WARNING: Overfitting Detection
print("\n" + "="*60)
print("OVERFITTING CHECK")
print("="*60)

for i, (train_r2, test_r2, degree) in enumerate(zip(train_scores, test_scores, degrees)):
    gap = train_r2 - test_r2
    if gap > 0.1:
        print(f"Degree {degree}: Possible overfitting (Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}, Gap = {gap:.4f})")
    else:
        print(f"✓ Degree {degree}: Good generalization (Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}, Gap = {gap:.4f})")