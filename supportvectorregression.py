from sklearn.metrics import r2_score,mean_squared_error
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


np.random.seed(42)
n = 400

rainfall_mm       = np.random.uniform(400, 1200, n)
soil_ph           = np.random.uniform(5.5, 8.0, n)
temperature_c     = np.random.uniform(22, 38, n)
fertilizer_kg     = np.random.uniform(30, 120, n)
irrigation_cycles = np.random.uniform(4, 20, n)
farm_age_years    = np.random.uniform(1, 30, n)
pesticide_litres  = np.random.uniform(0.5, 5.0, n)

noise = np.random.normal(0, 1.5, n)

yield_quintals = (
    15
    + 0.02  * rainfall_mm
    - 0.00001 * rainfall_mm**2
    + 0.3   * fertilizer_kg
    - 0.002 * fertilizer_kg**2
    - 3     * np.abs(soil_ph - 6.8)
    - 0.1   * np.abs(temperature_c - 28)
    - 0.5   * (irrigation_cycles * rainfall_mm / 1000)
    - 0.8   * pesticide_litres
    + noise
)

yield_quintals = np.clip(yield_quintals, 8, 45)

df = pd.DataFrame({
    'rainfall_mm'       : rainfall_mm,
    'soil_ph'           : soil_ph,
    'temperature_c'     : temperature_c,
    'fertilizer_kg'     : fertilizer_kg,
    'irrigation_cycles' : irrigation_cycles,
    'farm_age_years'    : farm_age_years,
    'pesticide_litres'  : pesticide_litres,
    'yield_quintals'    : yield_quintals
})



print(df.shape)
print(df.head())
print(df.describe())

X = df[['rainfall_mm','temperature_c','fertilizer_kg','irrigation_cycles','farm_age_years','pesticide_litres']]
y = df['yield_quintals']
print(X.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#model selection
LinearModel = LinearRegression()
LinearModel.fit(X_train,y_train)

#prediction for linear model
y_linear_prediction = LinearModel.predict(X_test)

y_linear_Train = LinearModel.predict(X_train)

#evaluation
print(f"MSE of linear model {mean_squared_error(y_test,y_linear_prediction)}")
print(f"R2_score of linear model test {r2_score(y_test,y_linear_prediction)}")

print(f"r2 score of train data {r2_score(y_train,y_linear_Train)}")

#plotting linear model

# plt.figure(figsize=(10,8))
# plt.scatter(range(len(y_test)), y_test, label='Actual data', color='blue')
# plt.plot(range(len(y_test)), y_linear_prediction, label='Predicted data', color='purple', linewidth=2)
# plt.xlabel('features of chilli')
# plt.ylabel('Predictions of yield_quintals')
# plt.legend()
# plt.show()

X_test_arr = X_test.values
y_test_arr = y_test.values

n_features = X_test_arr.shape[1]
feature_names = X_test.columns.tolist()  # grab names before converting

fig, axes = plt.subplots(nrows=n_features, ncols=1, figsize=(10, 4 * n_features))

for i, ax in enumerate(axes):
    sorted_idx = np.argsort(X_test_arr[:, i])

    ax.scatter(X_test_arr[:, i], y_test_arr, label='Actual', color='blue', alpha=0.7)
    ax.plot(X_test_arr[sorted_idx, i], y_linear_prediction[sorted_idx],
            label='Predicted', color='purple', linewidth=2)

    ax.set_xlabel(feature_names[i])
    ax.set_ylabel('yield_quintals')
    ax.set_title(f'SVR Prediction vs Actual — {feature_names[i]}')
    ax.legend()

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y_test, y_linear_prediction, color='blue', alpha=0.7, label='Predictions')
ax.plot([y_test.min(), y_test.max()], 
        [y_test.min(), y_test.max()], 
        color='purple', linewidth=2, label='Perfect fit line')

ax.set_xlabel('Actual yield_quintals')
ax.set_ylabel('Predicted yield_quintals')
ax.set_title('Linear Regression — Actual vs Predicted')
ax.legend()
plt.tight_layout()
plt.show()


#support vector regression

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Step 1 — Scale features (mandatory for SVR)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Step 2 — Define parameter grid
param_grid = {
    'C'      : [100, 500, 1000, 5000, 10000],  # push C much higher
    'gamma'  : ['scale', 0.001, 0.01, 0.05],   # fine-tune around 0.01
    'epsilon': [0.1, 0.5, 1, 2]
}

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, 
                           scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train_scaled)

print(f"Best parameters : {grid_search.best_params_}")
print(f"Best CV R²      : {grid_search.best_score_:.4f}")

best_svr = grid_search.best_estimator_
y_svr_scaled_pred = best_svr.predict(X_test_scaled)
y_svr_pred = scaler_y.inverse_transform(y_svr_scaled_pred.reshape(-1, 1)).ravel()

print(f"SVR MSE    : {mean_squared_error(y_test, y_svr_pred):.4f}")
print(f"SVR R² test: {r2_score(y_test, y_svr_pred):.4f}")
print(f"SVR R² train: {r2_score(y_train, scaler_y.inverse_transform(best_svr.predict(X_train_scaled).reshape(-1,1)).ravel()):.4f}")