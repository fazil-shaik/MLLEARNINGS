import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Setup Data Generation
np.random.seed(42)
n_samples = 100

sqft = np.random.randint(1000, 3500, n_samples)
lot = np.random.uniform(0.1, 2.0, n_samples)
year = np.random.randint(1980, 2024, n_samples)
dist = np.random.randint(1, 30, n_samples)

# Price formula with logic + noise
noise = np.random.normal(0, 50000, n_samples)
price = (sqft * 250) + (lot * 100000) + ((year - 1980) * 5000) - (dist * 3000) + noise
price = np.maximum(price, 50000) # Ensure no negative prices

df = pd.DataFrame({'sqft': sqft, 'lot': lot, 'year': year, 'Dist': dist, 'price': price.astype(int)})

# 2. Prepare Features and Target
X = df[['sqft', 'lot', 'year', 'Dist']]
y = df['price']

# 3. Train-Test Split (Crucial for avoiding data leaks!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scaling (Fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train k-NN
knn = KNeighborsRegressor(n_neighbors=5, weights='distance') # 'distance' helps with non-linear data
knn.fit(X_train_scaled, y_train)

# 6. Evaluation
y_train_pred = knn.predict(X_train_scaled)
y_test_pred = knn.predict(X_test_scaled)

print(f"Training R2 Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R2 Score: {r2_score(y_test, y_test_pred):.4f}")

# 7. Predict for a new house (MUST SCALE THE INPUT)
# Note: Year fixed from 11985 to 2020
new_house = np.array([[2500, 0.5, 2020, 5]]) 
new_house_scaled = scaler.transform(new_house) # Match the scaling of training data
prediction = knn.predict(new_house_scaled)
print(f"Predicted Price for the new house: ${prediction[0]:,.2f}")

# 8. Cross-Validation (Using all data but scaled correctly)
X_all_scaled = scaler.fit_transform(X)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X_all_scaled, y, cv=kf)
print(f"Cross-val R2 scores: {cv_scores}")
print(f"Mean CV R2: {cv_scores.mean():.4f}")

# 9. Visualizing Feature Importance
results = permutation_importance(knn, X_test_scaled, y_test, n_repeats=10, random_state=42)
sorted_idx = results.importances_mean.argsort()

plt.figure(figsize=(10, 5))
plt.barh(X.columns[sorted_idx], results.importances_mean[sorted_idx], color='skyblue')
plt.title("Feature Importance (Permutation)")
plt.xlabel("Decrease in R2 Score")
plt.show()

# 10. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='green', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Test Set)")
plt.show()