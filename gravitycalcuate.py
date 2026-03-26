import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000

mass = np.random.uniform(1e10, 1e26, n)           # planetary-scale masses
radius = np.random.uniform(1e5, 1e8, n)            # meters
density = mass / ((4/3) * np.pi * radius**3)       # derived
surface_area = 4 * np.pi * radius**2               # m²

noise = np.random.normal(1.0, 0.02, n)

# Target: surface gravity g = GM/r²  (G = 6.674e-11)
G = 6.674e-11
gravity = (G * mass / radius**2) * noise           # m/s²

df = pd.DataFrame({
    'mass_kg': mass,
    'radius_m': radius,
    'density_kg_m3': density,
    'surface_area_m2': surface_area,
    'surface_gravity_m_s2': gravity
})


df['log_mass']    = np.log(df['mass_kg'])
df['log_radius']  = np.log(df['radius_m'])
df['log_density'] = np.log(df['density_kg_m3'])
df['log_area']    = np.log(df['surface_area_m2'])
df['log_gravity'] = np.log(df['surface_gravity_m_s2'])

X = df[['log_mass', 'log_radius', 'log_density', 'log_area']]
y = df['log_gravity']   #  predicting log(gravity), not raw gravity




#linear regression train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#linear model
LinearModel = LinearRegression()
LinearModel.fit(X_train,y_train)


#predicting and eval
y_linear_predict = LinearModel.predict(X_test)

y_train_pred = LinearModel.predict(X_train)
y_test_pred  = LinearModel.predict(X_test)

r2_score_train = r2_score(y_train,y_train_pred)
r2_score_test  = r2_score(y_test,  y_test_pred)

print(f"r2 score of training data {r2_score_train}")
print(f"r2 score of testing data {r2_score_test}")

#eval
print(f"predictions are {y_linear_predict}")
print(f"r2_score of the model is {r2_score(y_test,y_linear_predict)}")





fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#plotting
axes[0].scatter(y_test, y_linear_predict, alpha=0.4, color='steelblue', edgecolors='white', linewidths=0.3, s=40, label='Predictions')

# Perfect prediction reference line
min_val = min(y_test.min(), y_linear_predict.min())
max_val = max(y_test.max(), y_linear_predict.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit (y = x)')

axes[0].set_xlabel('Actual log(gravity)', fontsize=12)
axes[0].set_ylabel('Predicted log(gravity)', fontsize=12)
axes[0].set_title(f'Predicted vs Actual (Log Space)\nR² = {r2_score(y_test, y_linear_predict):.4f}', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Residuals (errors) ---
residuals = y_test.values - y_linear_predict

axes[1].scatter(y_linear_predict, residuals, alpha=0.4, color='darkorange', edgecolors='white', linewidths=0.3, s=40)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero error line')

axes[1].set_xlabel('Predicted log(gravity)', fontsize=12)
axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
axes[1].set_title('Residual Plot\n(Should be random noise around 0)', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Gravity Prediction — Linear Regression in Log Space', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


planets = {
    "Mars-like":    (6.4e23,  3.4e6),
    "Jupiter-like": (1.9e27,  7.1e7),
    "Neutron Star": (2.0e30,  1.0e4),   # extreme edge case
    "Earth":        (5.97e24, 6.37e6),
}

print(f"\n{'Planet':<15} {'Predicted g':>12} {'Actual g':>12} {'Error %':>10}")
print("-" * 52)

for name, (m, r) in planets.items():
    d    = m / ((4/3) * np.pi * r**3)
    sa   = 4 * np.pi * r**2
    inp  = np.array([[np.log(m), np.log(r), np.log(d), np.log(sa)]])
    pred = np.exp(LinearModel.predict(inp)[0])
    true = G * m / r**2
    err  = abs(pred - true) / true * 100
    print(f"{name:<15} {pred:>12.4f} {true:>12.4f} {err:>9.4f}%")