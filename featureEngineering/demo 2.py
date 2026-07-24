import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, PowerTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

 
# 1. CREATE SYNTHETIC REAL-WORLD DATASET
 

def create_house_dataset(n_samples=1000):
    """Create a realistic house price dataset"""
    
    np.random.seed(42)
    
    # Base features
    sqft_living = np.random.normal(2000, 500, n_samples).clip(500, 5000)
    sqft_lot = np.random.normal(8000, 3000, n_samples).clip(1000, 20000)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.uniform(1, 4, n_samples).round(1)
    age = np.random.exponential(30, n_samples).clip(0, 100)
    
    # Categorical features
    waterfront = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    view = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    condition = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.4, 0.25, 0.1])
    grade = np.random.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12], n_samples, 
                            p=[0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.2, 0.1, 0.05, 0.02])
    
    # Location (zip code with price influence)
    zipcode = np.random.choice([98101, 98102, 98103, 98104, 98105, 98106, 98107, 98108], n_samples)
    zip_price_multiplier = {98101: 1.8, 98102: 1.6, 98103: 1.4, 98104: 1.2,
                           98105: 1.0, 98106: 0.8, 98107: 0.7, 98108: 0.6}
    
    # Create target variable (price) with realistic relationships
    base_price = 100000
    price = (base_price +
             sqft_living * 150 +
             sqft_lot * 0.5 +
             bedrooms * 15000 +
             bathrooms * 30000 -
             age * 800 +
             waterfront * 200000 +
             view * 30000 +
             condition * 15000 +
             grade * 25000)
    
    # Add zipcode multiplier
    for zip_code, multiplier in zip_price_multiplier.items():
        price[zipcode == zip_code] *= multiplier
    
    # Add non-linear relationship (interaction between sqft and grade)
    price += (sqft_living * grade) * 20
    
    # Add some noise
    noise = np.random.normal(0, 50000, n_samples)
    price = price + noise
    price = price.clip(100000, 2000000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'sqft_living': sqft_living.astype(int),
        'sqft_lot': sqft_lot.astype(int),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age.astype(int),
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'zipcode': zipcode,
        'price': price.astype(int)
    })
    
    # Add missing values (10% missing in some columns)
    for col in ['bedrooms', 'bathrooms', 'view']:
        missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df

# Generate the dataset
df = create_house_dataset(1000)
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

 
# 2. FIXED FEATURE ENGINEERING PIPELINE
 

class FixedFeatureEngineeringPipeline:
    """Complete feature engineering pipeline with proper NaN handling"""
    
    def __init__(self, df, target_col='price'):
        self.df = df.copy()
        self.target_col = target_col
        self.X = self.df.drop(columns=[target_col])
        self.y = self.df[target_col]
        
    def handle_missing_values_safe(self):
        """Handle missing values safely without creating new NaNs"""
        print("\n--- Handling Missing Values ---")
        
        # Fill missing values using SimpleImputer
        from sklearn.impute import SimpleImputer
        
        # Separate columns by type
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use median for numeric columns (more robust than mean)
        numeric_imputer = SimpleImputer(strategy='median')
        self.X[numeric_cols] = numeric_imputer.fit_transform(self.X[numeric_cols])
        
        print(f"Missing values after imputation: {self.X.isnull().sum().sum()}")
        print(f"NaN count in each column:\n{self.X.isnull().sum()}")
        
        return self
    
    def create_interaction_features_safe(self):
        """Create interaction features with safe division"""
        print("\n--- Creating Interaction Features ---")
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        
        # Ratio features with safe division
        self.X['price_per_sqft'] = self.y / (self.X['sqft_living'] + epsilon)
        self.X['bathroom_per_bedroom'] = self.X['bathrooms'] / (self.X['bedrooms'] + epsilon)
        self.X['lot_to_living_ratio'] = self.X['sqft_lot'] / (self.X['sqft_living'] + epsilon)
        
        # Interaction terms
        self.X['sqft_grade_interaction'] = self.X['sqft_living'] * self.X['grade']
        self.X['age_condition_interaction'] = self.X['age'] * self.X['condition']
        
        # Polynomial features
        self.X['sqft_squared'] = self.X['sqft_living'] ** 2
        self.X['bathrooms_squared'] = self.X['bathrooms'] ** 2
        
        # Check for any NaNs created
        print(f"NaN count after interaction features: {self.X.isnull().sum().sum()}")
        
        return self
    
    def encode_categorical_features_safe(self):
        """Encode categorical features safely"""
        print("\n--- Encoding Categorical Features ---")
        
        # One-hot encoding for zipcode
        zipcode_dummies = pd.get_dummies(self.X['zipcode'], prefix='zip', dummy_na=False)
        self.X = pd.concat([self.X, zipcode_dummies], axis=1)
        self.X.drop('zipcode', axis=1, inplace=True)
        
        print(f"NaN count after encoding: {self.X.isnull().sum().sum()}")
        
        return self
    
    def transform_numerical_features_safe(self):
        """Apply transformations with NaN checking"""
        print("\n--- Transforming Numerical Features ---")
        
        # Log transformation (add 1 to avoid log(0))
        skewed_features = ['sqft_lot', 'age']
        for feature in skewed_features:
            # Add small value to avoid log(0) and ensure positivity
            self.X[f'{feature}_log'] = np.log1p(self.X[feature].clip(lower=0))
            self.X.drop(feature, axis=1, inplace=True)
        
        # Square root transformation (clip negative to 0)
        self.X['sqft_living_sqrt'] = np.sqrt(self.X['sqft_living'].clip(lower=0))
        
        # Log transform target variable
        self.y_transformed = np.log1p(self.y.clip(lower=0))
        
        # Final NaN check
        print(f"NaN count after transformations: {self.X.isnull().sum().sum()}")
        
        return self
    
    def create_domain_features_safe(self):
        """Create domain-specific features safely"""
        print("\n--- Creating Domain-Specific Features ---")
        
        # House quality score
        self.X['quality_score'] = self.X['grade'] * self.X['condition']
        
        # Is premium property?
        self.X['is_premium'] = ((self.X['grade'] >= 9) & (self.X['waterfront'] == 1)).astype(int)
        
        # Age category - ensure no NaN after binning
        self.X['age_category'] = pd.cut(self.X['age'], 
                                        bins=[-1, 5, 20, 50, 100],
                                        labels=['new', 'recent', 'old', 'historic'])
        
        # Fill any potential NaN from binning
        self.X['age_category'] = self.X['age_category'].fillna('historic')
        
        # One-hot encode age category
        age_dummies = pd.get_dummies(self.X['age_category'], prefix='age')
        self.X = pd.concat([self.X, age_dummies], axis=1)
        self.X.drop('age_category', axis=1, inplace=True)
        
        print(f"NaN count after domain features: {self.X.isnull().sum().sum()}")
        
        return self
    
    def final_nan_cleanup(self):
        """Final cleanup - ensure absolutely no NaNs remain"""
        print("\n--- Final NaN Cleanup ---")
        
        # Check for any remaining NaNs
        if self.X.isnull().sum().sum() > 0:
            print(f"Found {self.X.isnull().sum().sum()} remaining NaNs. Filling with 0...")
            self.X = self.X.fillna(0)
        
        # Also check for inf values
        self.X = self.X.replace([np.inf, -np.inf], 0)
        
        print(f"Final NaN count: {self.X.isnull().sum().sum()}")
        print(f"Final shape: {self.X.shape}")
        
        return self
    
    def scale_features(self):
        """Scale numerical features"""
        print("\n--- Scaling Features ---")
        
        # Select only numeric columns (all should be numeric now)
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use RobustScaler (handles outliers better)
        self.scaler = RobustScaler()
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X[numerical_cols]),
            columns=numerical_cols,
            index=self.X.index
        )
        
        print(f"Scaled {len(numerical_cols)} features")
        
        return self
    
    def select_features(self, n_features=20):
        """Feature selection using multiple methods"""
        print("\n--- Feature Selection ---")
        
        X = self.X_scaled
        y = self.y_transformed
        
        # Method 1: Correlation with target
        correlations = X.corrwith(pd.Series(y, index=X.index)).abs().sort_values(ascending=False)
        top_corr = correlations.head(n_features).index.tolist()
        
        # Method 2: Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top_rf = importances.nlargest(n_features).index.tolist()
        
        # Combine both methods
        selected_features = list(set(top_corr + top_rf))
        
        print(f"Selected {len(selected_features)} features from {len(X.columns)} total")
        print(f"Top 5 features by correlation:\n{list(top_corr[:5])}")
        print(f"\nTop 5 features by RF importance:\n{list(top_rf[:5])}")
        
        return X[selected_features]
    
    def run_full_pipeline(self):
        """Execute complete feature engineering pipeline with safety checks"""
        print("="*50)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("="*50)
        
        # Apply all transformations with safety checks
        self.handle_missing_values_safe()
        self.encode_categorical_features_safe()
        self.create_interaction_features_safe()
        self.create_domain_features_safe()
        self.transform_numerical_features_safe()
        self.final_nan_cleanup()  # Critical step!
        self.scale_features()
        
        # Select best features
        self.X_final = self.select_features(n_features=20)
        
        # Final validation
        assert not self.X_final.isnull().any().any(), "Still have NaN values!"
        assert not np.isinf(self.X_final).any().any(), "Still have inf values!"
        
        print("\n" + "="*50)
        print("FEATURE ENGINEERING COMPLETE - NO NaNs REMAINING")
        print(f"Final dataset shape: {self.X_final.shape}")
        print("="*50)
        
        return self.X_final, self.y_transformed

# Run the fixed pipeline
print("\n" + "="*50)
print("RUNNING FIXED FEATURE ENGINEERING PIPELINE")
print("="*50)

feature_engineer = FixedFeatureEngineeringPipeline(df)
X_engineered, y_engineered = feature_engineer.run_full_pipeline()

# Verify no NaNs remain
print(f"\nVERIFICATION: Any NaN in X_engineered? {X_engineered.isnull().any().any()}")
print(f"VERIFICATION: Any NaN in y_engineered? {np.isnan(y_engineered).any()}")

 
# 3. TRAIN-TEST SPLIT
 

print("\n" + "="*50)
print("TRAIN-TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y_engineered, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

 
# 4. MODEL COMPARISON (FIXED)
 

print("\n" + "="*50)
print("MODEL PERFORMANCE COMPARISON")
print("="*50)

# Model 1: Simple Linear Regression without feature engineering
print("\n--- Baseline Model (Raw Features, No Engineering) ---")

# Prepare raw features
X_raw = df.drop('price', axis=1)
X_raw = X_raw.select_dtypes(include=[np.number])

# Handle NaNs in raw data
from sklearn.impute import SimpleImputer
raw_imputer = SimpleImputer(strategy='median')
X_raw_imputed = raw_imputer.fit_transform(X_raw)
y_raw = np.log1p(df['price'])

X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
    X_raw_imputed, y_raw, test_size=0.2, random_state=42
)

lr_raw = LinearRegression()
lr_raw.fit(X_raw_train, y_raw_train)
y_raw_pred = lr_raw.predict(X_raw_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_raw_test, y_raw_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_raw_test, y_raw_pred):.2f}")
print(f"R² Score: {r2_score(y_raw_test, y_raw_pred):.4f}")

# Model 2: Linear Regression with Feature Engineering (NOW WORKS!)
print("\n--- Linear Regression with Feature Engineering ---")
lr = LinearRegression()
lr.fit(X_train, y_train)  # This should now work without NaN errors
y_pred = lr.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Model 3: Ridge Regression (Regularized)
print("\n--- Ridge Regression with Feature Engineering ---")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_ridge):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_ridge):.4f}")

# Model 4: Random Forest (handles complex interactions)
print("\n--- Random Forest with Feature Engineering ---")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_rf):.4f}")

 
# 5. FEATURE IMPORTANCE ANALYSIS
 

print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

feature_importance = pd.DataFrame({
    'feature': X_engineered.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

 
# 6. VISUALIZATIONS
 

plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted (Ridge)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ridge, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title(f'Ridge Regression\nR² = {r2_score(y_test, y_pred_ridge):.4f}')

# Plot 2: Actual vs Predicted (Random Forest)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title(f'Random Forest\nR² = {r2_score(y_test, y_pred_rf):.4f}')

plt.tight_layout()
plt.show()

# Plot 3: Feature Importance Bar Chart
plt.figure(figsize=(10, 8))
plt.barh(feature_importance.head(10)['feature'], 
         feature_importance.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

 
# 7. RESIDUAL ANALYSIS
 

print("\n" + "="*50)
print("RESIDUAL ANALYSIS")
print("="*50)

residuals = y_test - y_pred_rf

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')

plt.subplot(1, 3, 2)
plt.scatter(y_pred_rf, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')

plt.subplot(1, 3, 3)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
plt.show()

print(f"\nResidual Statistics:")
print(f"Mean: {residuals.mean():.4f}")
print(f"Std: {residuals.std():.4f}")
print(f"Skewness: {residuals.skew():.4f}")

 
# 8. CROSS-VALIDATION RESULTS
 

print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS (5-Fold)")
print("="*50)

cv_scores_rf = cross_val_score(rf, X_engineered, y_engineered, 
                               cv=5, scoring='r2')
print(f"Random Forest - R² scores: {cv_scores_rf}")
print(f"Mean R²: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

cv_scores_ridge = cross_val_score(ridge, X_engineered, y_engineered, 
                                  cv=5, scoring='r2')
print(f"\nRidge Regression - R² scores: {cv_scores_ridge}")
print(f"Mean R²: {cv_scores_ridge.mean():.4f} (+/- {cv_scores_ridge.std() * 2:.4f})")

 
# 9. PERFORMANCE SUMMARY
 

print("\n" + "="*50)
print("PERFORMANCE SUMMARY")
print("="*50)

summary_df = pd.DataFrame({
    'Model': ['Baseline (No Engineering)', 'Linear Regression', 'Ridge Regression', 'Random Forest'],
    'R² Score': [
        r2_score(y_raw_test, y_raw_pred),
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred_ridge),
        r2_score(y_test, y_pred_rf)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_raw_test, y_raw_pred)),
        np.sqrt(mean_squared_error(y_test, y_pred)),
        np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf))
    ]
})
print(summary_df)