# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           explained_variance_score)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

 
# PART 1: Synthetic Dataset Creation
 

print("="*80)
print("SVM vs GRADIENT BOOSTING REGRESSION COMPARISON")
print("="*80)

np.random.seed(42)
n_samples = 1000

# Create complex dataset with multiple patterns
X = np.random.uniform(-3, 3, n_samples)

# Create target with multiple patterns
y = (np.sin(2 * X) +                        # Periodic pattern
     0.5 * X**2 +                            # Quadratic pattern
     0.3 * np.random.normal(0, 1, n_samples) + # Noise
     2 * np.where(X > 1, X, 0) +              # Threshold pattern
     np.exp(-X**2) * 2)                       # Bell curve

# Add some outliers
outlier_idx = np.random.choice(n_samples, 20, replace=False)
y[outlier_idx] += np.random.normal(5, 2, 20)

# Reshape for sklearn
X = X.reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for SVM)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

print(f"\nDataset Info:")
print(f"Total samples: {n_samples}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}")

 
# PART 2: Visualize Data Patterns
 

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Original data
axes[0].scatter(X, y, alpha=0.5, s=30, label='Data points')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('Original Data Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Training vs Test split
axes[1].scatter(X_train, y_train, alpha=0.5, s=20, label='Training', c='blue')
axes[1].scatter(X_test, y_test, alpha=0.5, s=20, label='Test', c='red')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title('Train-Test Split')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

 
# PART 3: Train Both Models
 

# 3.1 SVM Model
print("\n" + "="*60)
print("TRAINING SVM REGRESSOR")
print("="*60)

svm_model = SVR(
    kernel='rbf',
    C=100,
    epsilon=0.1,
    gamma='scale'
)

svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

# 3.2 Gradient Boosting Model
print("\n" + "="*60)
print("TRAINING GRADIENT BOOSTING REGRESSOR")
print("="*60)

gbr_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

gbr_model.fit(X_train, y_train)
gbr_pred = gbr_model.predict(X_test)

 
# PART 4: Performance Metrics
 

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate various regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Explained Var': ev,
        'MAPE (%)': mape
    }

# Calculate metrics
svm_metrics = calculate_metrics(y_test, svm_pred, 'SVM')
gbr_metrics = calculate_metrics(y_test, gbr_pred, 'Gradient Boosting')

# Create comparison dataframe
comparison_df = pd.DataFrame([svm_metrics, gbr_metrics])
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

 
# PART 5: Detailed Visualization
 

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Sort X values for smooth line plotting
X_sorted = np.sort(X_test, axis=0)
X_sorted_scaled = scaler_X.transform(X_sorted)
y_sorted = y_test[np.argsort(X_test.flatten())]

# 1. SVM Predictions
axes[0, 0].scatter(X_test, y_test, alpha=0.4, s=20, label='Actual', c='gray')
axes[0, 0].plot(X_sorted, svm_model.predict(X_sorted_scaled), 
                'r-', lw=2, label='SVM Prediction')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title('SVM Regression Predictions')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Gradient Boosting Predictions
axes[0, 1].scatter(X_test, y_test, alpha=0.4, s=20, label='Actual', c='gray')
axes[0, 1].plot(X_sorted, gbr_model.predict(X_sorted), 
                'b-', lw=2, label='GBR Prediction')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('y')
axes[0, 1].set_title('Gradient Boosting Regression Predictions')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Both Models Comparison
axes[0, 2].scatter(X_test, y_test, alpha=0.3, s=15, label='Actual', c='gray')
axes[0, 2].plot(X_sorted, svm_model.predict(X_sorted_scaled), 
                'r-', lw=2, label='SVM', alpha=0.8)
axes[0, 2].plot(X_sorted, gbr_model.predict(X_sorted), 
                'b-', lw=2, label='GBR', alpha=0.8)
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('y')
axes[0, 2].set_title('Models Comparison')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Residuals Comparison
svm_residuals = y_test - svm_pred
gbr_residuals = y_test - gbr_pred

axes[1, 0].scatter(svm_pred, svm_residuals, alpha=0.5, s=30, 
                   c='red', label='SVM', edgecolors='black', linewidth=0.5)
axes[1, 0].axhline(y=0, color='black', linestyle='--', lw=1)
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residual Plot - SVM')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(gbr_pred, gbr_residuals, alpha=0.5, s=30, 
                   c='blue', label='GBR', edgecolors='black', linewidth=0.5)
axes[1, 1].axhline(y=0, color='black', linestyle='--', lw=1)
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residual Plot - Gradient Boosting')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Error Distribution
axes[1, 2].hist(svm_residuals, bins=30, alpha=0.5, label='SVM', 
                edgecolor='black', color='red')
axes[1, 2].hist(gbr_residuals, bins=30, alpha=0.5, label='GBR', 
                edgecolor='black', color='blue')
axes[1, 2].axvline(x=0, color='black', linestyle='--', lw=1)
axes[1, 2].set_xlabel('Residuals')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Error Distribution Comparison')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

 
# PART 6: Cross-Validation Comparison
 

print("\n" + "="*60)
print("CROSS-VALIDATION COMPARISON (5-Fold)")
print("="*60)

# SVM Cross-validation
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, 
                                cv=5, scoring='r2')
print(f"\nSVM CV R² scores: {svm_cv_scores}")
print(f"SVM CV R² mean: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std() * 2:.4f})")

# GBR Cross-validation
gbr_cv_scores = cross_val_score(gbr_model, X_train, y_train, 
                                cv=5, scoring='r2')
print(f"\nGBR CV R² scores: {gbr_cv_scores}")
print(f"GBR CV R² mean: {gbr_cv_scores.mean():.4f} (+/- {gbr_cv_scores.std() * 2:.4f})")

 
# PART 7: Learning Curves
 

from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, title, ax, color):
    """Plot learning curves for an estimator"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color=color)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                    alpha=0.1, color=color)
    ax.plot(train_sizes, train_mean, 'o-', color=color, label='Training score')
    ax.plot(train_sizes, test_mean, 'o-', color=color, linestyle='--', 
            label='Cross-validation score')
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('R² Score')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

plot_learning_curves(svm_model, X_train_scaled, y_train, 
                     'SVM Learning Curves', axes[0], 'red')
plot_learning_curves(gbr_model, X_train, y_train, 
                     'Gradient Boosting Learning Curves', axes[1], 'blue')

plt.tight_layout()
plt.show()

 
# PART 8: Real-World Example - Boston Housing Dataset
 

print("\n" + "="*60)
print("REAL-WORLD EXAMPLE: Boston Housing Dataset")
print("="*60)

from sklearn.datasets import fetch_california_housing

# Load California housing dataset (similar to Boston)
housing = fetch_california_housing()
X_housing, y_housing = housing.data, housing.target

# Use only first 2 features for visualization
X_housing_2d = X_housing[:, :2]

# Split
X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_housing_2d, y_housing, test_size=0.2, random_state=42
)

# Scale for SVM
scaler_h = StandardScaler()
X_h_train_scaled = scaler_h.fit_transform(X_h_train)
X_h_test_scaled = scaler_h.transform(X_h_test)

# Train models on housing data
svm_housing = SVR(kernel='rbf', C=100, epsilon=0.1)
svm_housing.fit(X_h_train_scaled, y_h_train)
svm_h_pred = svm_housing.predict(X_h_test_scaled)

gbr_housing = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbr_housing.fit(X_h_train, y_h_train)
gbr_h_pred = gbr_housing.predict(X_h_test)

# Calculate metrics
print("\nHousing Dataset Results:")
print("-" * 40)
print(f"SVM R² Score: {r2_score(y_h_test, svm_h_pred):.4f}")
print(f"GBR R² Score: {r2_score(y_h_test, gbr_h_pred):.4f}")
print(f"\nSVM RMSE: ${np.sqrt(mean_squared_error(y_h_test, svm_h_pred)) * 100000:,.0f}")
print(f"GBR RMSE: ${np.sqrt(mean_squared_error(y_h_test, gbr_h_pred)) * 100000:,.0f}")

 
# PART 9: Hyperparameter Impact
 

print("\n" + "="*60)
print("HYPERPARAMETER IMPACT ANALYSIS")
print("="*60)

# SVM Parameter impact
C_values = [0.1, 1, 10, 100, 1000, 10000]
svm_scores = []

print("\nSVM - Impact of C (regularization):")
for C in C_values:
    svm_temp = SVR(kernel='rbf', C=C, epsilon=0.1)
    svm_temp.fit(X_train_scaled, y_train)
    score = r2_score(y_test, svm_temp.predict(X_test_scaled))
    svm_scores.append(score)
    print(f"C = {C:6d} -> R² = {score:.4f}")

# GBR Parameter impact
n_estimators = [10, 50, 100, 200, 500]
gbr_scores = []

print("\nGBR - Impact of n_estimators:")
for n in n_estimators:
    gbr_temp = GradientBoostingRegressor(n_estimators=n, max_depth=3, random_state=42)
    gbr_temp.fit(X_train, y_train)
    score = r2_score(y_test, gbr_temp.predict(X_test))
    gbr_scores.append(score)
    print(f"n_estimators = {n:3d} -> R² = {score:.4f}")

# Visualize parameter impact
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].semilogx(C_values, svm_scores, 'ro-', linewidth=2, markersize=8)
axes[0].set_xlabel('C (Regularization Parameter)')
axes[0].set_ylabel('R² Score')
axes[0].set_title('SVM Performance vs C Parameter')
axes[0].grid(True, alpha=0.3)

axes[1].plot(n_estimators, gbr_scores, 'bo-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Estimators')
axes[1].set_ylabel('R² Score')
axes[1].set_title('GBR Performance vs n_estimators')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

 
# PART 10: Robustness to Outliers
 

# Add synthetic outliers to test robustness
X_outliers = np.vstack([X, [[4.5], [4.8], [5.0]]])
y_outliers = np.hstack([y, [15, 18, 20]])  # Extreme values

# Split outlier data
X_out_train, X_out_test, y_out_train, y_out_test = train_test_split(
    X_outliers, y_outliers, test_size=0.2, random_state=42
)

# Scale
scaler_out = StandardScaler()
X_out_train_scaled = scaler_out.fit_transform(X_out_train)
X_out_test_scaled = scaler_out.transform(X_out_test)

# Train models with outliers
svm_out = SVR(kernel='rbf', C=100, epsilon=0.1)
svm_out.fit(X_out_train_scaled, y_out_train)
svm_out_pred = svm_out.predict(X_out_test_scaled)

gbr_out = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbr_out.fit(X_out_train, y_out_train)
gbr_out_pred = gbr_out.predict(X_out_test)

print("\n" + "="*60)
print("ROBUSTNESS TO OUTLIERS TEST")
print("="*60)
print("\nPerformance with outliers added:")
print(f"SVM R²: {r2_score(y_out_test, svm_out_pred):.4f}")
print(f"GBR R²: {r2_score(y_out_test, gbr_out_pred):.4f}")

 
# PART 11: Comprehensive Comparison Table
 

print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON: SVM vs GRADIENT BOOSTING")
print("="*80)

comparison_points = {
    'Aspect': [
        'Best for',
        'Data size',
        'Training speed',
        'Prediction speed',
        'Handling non-linearity',
        'Feature scaling needed',
        'Interpretability',
        'Robustness to outliers',
        'Memory usage',
        'Hyperparameter tuning',
        'Probability estimates',
        'Handling categorical features'
    ],
    'SVM': [
        'Complex boundaries, clear margins',
        'Small to medium datasets',
        'Slow (O(n²) or O(n³))',
        'Fast',
        'Excellent (kernel trick)',
        'Yes - crucial',
        'Low (black box)',
        'High (ε-insensitive)',
        'High (support vectors)',
        'Moderate (C, ε, γ)',
        'No direct probability',
        'Requires encoding'
    ],
    'Gradient Boosting': [
        'Complex patterns, large data',
        'Small to very large',
        'Moderate to slow (sequential)',
        'Fast',
        'Excellent (ensemble)',
        'Not necessary',
        'Medium (feature importance)',
        'Low (can overfit outliers)',
        'Moderate (many trees)',
        'High (many parameters)',
        'No direct probability',
        'Handles natively'
    ]
}

comparison_df = pd.DataFrame(comparison_points)
print("\n", comparison_df.to_string(index=False))

 
# PART 12: When to Use Each Model
 

print("\n" + "="*80)
print("PRACTICAL GUIDELINES")
print("="*80)

print("""
📊 USE SVM WHEN:
   • You have small to medium datasets (<10,000 samples)
   • You need robustness to outliers
   • Data is well-scaled and preprocessed
   • You want smooth, flexible decision boundaries
   • You have clear margins/separation in data
   • Memory isn't a huge constraint

🎯 USE GRADIENT BOOSTING WHEN:
   • You have large datasets (10,000+ samples)
   • You want feature importance insights
   • You have mixed data types (numerical + categorical)
   • You need quick prototyping
   • You're competing in ML competitions (often wins!)
   • You want to handle complex interactions automatically

⚠️ PRO TIPS:
   1. Always scale features for SVM
   2. Start with RBF kernel for SVM
   3. For GBR, monitor overfitting with validation
   4. Ensemble both? Sometimes they complement each other!
""")



# Create mesh for decision boundary visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = y.min() - 1, y.max() + 1
xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)

# Get predictions for mesh
xx_scaled = scaler_X.transform(xx)
svm_xx_pred = svm_model.predict(xx_scaled)
gbr_xx_pred = gbr_model.predict(xx)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# SVM decision boundary
axes[0].scatter(X, y, alpha=0.3, s=20, c='gray', label='Training data')
axes[0].plot(xx, svm_xx_pred, 'r-', lw=2, label='SVM prediction')
axes[0].fill_between(xx.flatten(), 
                     svm_xx_pred - svm_model.epsilon, 
                     svm_xx_pred + svm_model.epsilon, 
                     alpha=0.2, color='red', label='ε-tube')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('SVM: ε-tube and Decision Boundary')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gradient Boosting decision boundary
axes[1].scatter(X, y, alpha=0.3, s=20, c='gray', label='Training data')
axes[1].plot(xx, gbr_xx_pred, 'b-', lw=2, label='GBR prediction')
# Add individual tree predictions (first 10 trees)
for i, pred in enumerate(gbr_model.estimators_[:10, 0]):
    axes[1].plot(xx, pred.predict(xx), 'b--', alpha=0.2, linewidth=0.5)
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title('GBR: Ensemble of Trees')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("COMPARISON COMPLETE - CHOOSE WISELY! 🎯")
print("="*80)