import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge

np.random.seed(42)
n = 1000

mass = np.random.uniform(1e10, 1e26, n)
radius = np.random.uniform(1e5, 1e8, n)
density = mass / ((4/3) * np.pi * radius**3)
surface_area = 4 * np.pi * radius**2
noise = np.random.normal(1.0, 0.3, n)

G = 6.674e-11
gravity = (G * mass / radius**2) * noise

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


# Add 10 fake noise columns to df
for i in range(10):
    df[f'noise_{i}'] = np.random.randn(n)

# Build X with ALL 14 features
real_features  = ['log_mass', 'log_radius', 'log_density', 'log_area']
noise_features = [f'noise_{i}' for i in range(10)]

X = df[real_features + noise_features]   # ← 14 columns total
y = df['log_gravity']

feature_names = X.columns.tolist()  # Save feature names for later reference
# X = df[['log_mass', 'log_radius', 'log_density', 'log_area']]
# y = df['log_gravity']   # predicting log(gravity), not raw gravity

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_log_pred = model.predict(X_test)

y_pred_actual = np.exp(y_log_pred)
y_test_actual = np.exp(y_test)

y_test_prediction = model.predict(X_test)
y_train_prediction = model.predict(X_train)

print(f"r2 score of test set: {r2_score(y_test, y_test_prediction):.6f}")
print(f"r2 score of train set: {r2_score(y_train, y_train_prediction):.6f}")

cv2_score = r2_score(y_test, y_log_pred)
print(f"Cross-validated R² (log space): {cv2_score:.6f}")
cv2_score_actual = r2_score(y_test_actual, y_pred_actual)
print(f"Cross-validated R² (actual gravity): {cv2_score_actual:.6f}")

print(f"Number of features: {X.shape[1]}")   # should say 14
print(model.coef_)                            # should show 14 coefficients



# print("=== Log-space R² (how well model fits log relationship) ===")
# print(f"R² (log space) : {r2_score(y_test, y_log_pred):.6f}")

# print("\n=== Actual gravity space R² ===")
# print(f"R² (actual)    : {r2_score(y_test_actual, y_pred_actual):.6f}")

# print("\n=== What the model learned ===")
# for feat, coef in zip(X.columns, model.coef_):
#     print(f"  {feat:15s} → coefficient: {coef:.4f}")
# print(f"  {'intercept':15s} → {model.intercept_:.4f}")

# print("\n=== Physics check ===")
# print("Expected: log_mass coef ≈ +1.0, log_radius coef ≈ -2.0")
# print("(because g = G·M·r⁻²  →  log(g) = log(G) + 1·log(M) - 2·log(r))")


# plt.style.use('dark_background')
# fig,axes = plt.subplots(1,2,figsize=(12,5))
# axes[0].scatter(y_test_actual,y_pred_actual,alpha=0.5,color='blue')
# axes[0].plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
# axes[0].set_xlabel('Actual Gravity (m/s²)')
# axes[0].set_ylabel('Predicted Gravity (m/s²)')
# axes[0].set_title('Actual vs Predicted Gravity')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
# # axes[0].grid(True, which="both", ls="--", linewidth=0.5)
# axes[0].set_aspect('equal', adjustable='box')
# axes[0].legend(['Perfect Prediction', 'Predicted vs Actual'], loc='upper left')
# axes[1].scatter(y_test, y_log_pred, alpha=0.5, color='green')
# axes[1].plot([y_test.min(), y_test.max()], [y_test.min(),
#     y_test.max()], 'r--')   
# axes[1].set_xlabel('Actual log(Gravity)')
# axes[1].set_ylabel('Predicted log(Gravity)')
# axes[1].set_title('Actual vs Predicted log(Gravity)')
# axes[1].grid(True, which="both", ls="--", linewidth=0.5)
# axes[1].set_aspect('equal', adjustable='box')
# axes[1].legend(['Perfect Prediction', 'Predicted vs Actual'], loc='upper left')
# plt.savefig('linear_model_performance.png', dpi=300)
# plt.tight_layout()
# plt.show()


# poly_model = PolynomialFeatures(degree=6,include_bias=False)
# X_poly = poly_model.fit_transform(X_train)

# Model = LinearRegression()
# Model.fit(X_poly, y_train)

# X_test_poly = poly_model.transform(X_test)
# y_poly_pred = Model.predict(X_test_poly)

# y_poly_pred_actual = np.exp(y_poly_pred)
# print("\n=== Polynomial Model R² (actual gravity) ===")
# print(f"R² (actual)    : {r2_score(y_test_actual, y_poly_pred_actual):.6f}")

# plt.style.use('dark_background')
# fig, axes = plt.subplots(1,3,figsize=(12,5))
# axes[0].scatter(y_test_actual,y_poly_pred_actual,alpha=0.5,color='cyan')
# axes[0].plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
# axes[0].set_xlabel('Actual Gravity (m/s²)')
# axes[0].set_ylabel('Predicted Gravity (m/s²)')
# axes[0].set_title('Polynomial Model: Actual vs Predicted Gravity')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
# axes[0].set_aspect('equal', adjustable='box')
# axes[0].legend(['Perfect Prediction', 'Predicted vs Actual'], loc='upper left')
# axes[1].scatter(y_test, y_poly_pred, alpha=0.5, color       ='magenta')
# axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')   
# axes[1].set_xlabel('Actual log(Gravity)')
# axes[1].set_ylabel('Predicted log(Gravity)')
# axes[1].set_title('Polynomial Model: Actual vs Predicted log(Gravity)')
# axes[1].grid(True, which="both", ls="--", linewidth=0.5)
# axes[1].set_aspect('equal', adjustable='box')
# axes[1].legend(['Perfect Prediction', 'Predicted vs Actual'], loc='upper left')
# plt.savefig('polynomial_model_performance.png', dpi=300)
# plt.tight_layout()
# plt.show()

# plt.style.use('dark_background')
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # --- Plot 1: Actual vs Predicted (log scale dots) ---
# axes[0].scatter(y_test_actual, y_poly_pred_actual, alpha=0.5, color='cyan', s=20)
# axes[0].plot([y_test_actual.min(), y_test_actual.max()],
#              [y_test_actual.min(), y_test_actual.max()], 'r--', linewidth=2)
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
# axes[0].set_xlabel('Actual Gravity (m/s²)')
# axes[0].set_ylabel('Predicted Gravity (m/s²)')
# axes[0].set_title('Actual vs Predicted (Log Scale)')
# axes[0].legend(['Perfect fit', 'Predictions'])
# axes[0].grid(True, which="both", ls="--", linewidth=0.3, alpha=0.5)

# # --- Plot 2: SORTED LINE PLOT — this gives you the curve ---
# sort_idx = np.argsort(y_test_actual.values)           # sort by actual gravity
# sorted_actual = y_test_actual.values[sort_idx]
# sorted_pred   = y_poly_pred_actual[sort_idx]

# axes[1].plot(sorted_actual, color='red',     linewidth=1.5, label='Actual Gravity')
# axes[1].plot(sorted_pred,   color='cyan',    linewidth=1.5, label='Predicted Gravity', linestyle='--')
# axes[1].set_xlabel('Samples (sorted by actual gravity)')
# axes[1].set_ylabel('Gravity (m/s²)')
# axes[1].set_title('Sorted: Actual vs Predicted Curve')  # ← THIS gives the curve
# axes[1].legend()
# axes[1].grid(True, ls="--", linewidth=0.3, alpha=0.5)

# # --- Plot 3: Residuals ---
# residuals = y_test_actual.values - y_poly_pred_actual
# axes[2].scatter(y_poly_pred_actual, residuals, alpha=0.4, color='magenta', s=20)
# axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
# axes[2].set_xlabel('Predicted Gravity (m/s²)')
# axes[2].set_ylabel('Residual')
# axes[2].set_title('Residual Plot')
# axes[2].grid(True, ls="--", linewidth=0.3, alpha=0.5)

# plt.suptitle('Polynomial Model Performance', fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.savefig('polynomial_model_performance.png', dpi=300, bbox_inches='tight')
# plt.show()




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


ridege_model = Ridge(alpha=1.0)
ridege_model.fit(X_scaled,y_train)

lasso_model = Lasso(alpha=0.0001,max_iter=10000)
lasso_model.fit(X_scaled,y_train)

y_ridge_pred = ridege_model.predict(X_test_scaled)
y_lasso_pred = lasso_model.predict(X_test_scaled)

y_ridge_pred_actual = np.exp(y_ridge_pred)
y_lasso_pred_actual = np.exp(y_lasso_pred)

ridge_log_pred = ridege_model.predict(X_test_scaled)
lasso_log_pred = lasso_model.predict(X_test_scaled)

# ── Convert back to actual gravity ────────────────────────────────
ridge_actual_pred = np.exp(ridge_log_pred)
lasso_actual_pred = np.exp(lasso_log_pred)
y_test_actual     = np.exp(y_test)


print("\n=== Ridge Regression R² (actual gravity) ===")
print(f"R² (actual)    : {r2_score(y_test_actual, y_ridge_pred_actual):.6f}")
print("\n=== Lasso Regression R² (actual gravity) ===")
print(f"R² (actual)    : {r2_score(y_test_actual, y_lasso_pred_actual):.6f}")

print(f"Ridge R² (actual gravity): {r2_score(y_test_actual, y_ridge_pred_actual):.6f}")
print(f"Lasso R² (actual gravity): {r2_score(y_test_actual, y_lasso_pred_actual):.6f}")

# ── Coefficient comparison table ───────────────────────────────────
print(f"\n{'Feature':<15} {'LinearReg':>12} {'Ridge':>12} {'Lasso':>12}")
print("-" * 54)
for name, lr_c, ri_c, la_c in zip(feature_names, model.coef_, ridege_model.coef_, lasso_model.coef_):
    print(f"{name:<15} {lr_c:>12.6f} {ri_c:>12.6f} {la_c:>12.6f}")



import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f0f0f')

colors = {
    'LinearReg': '#4fc3f7',
    'Ridge':     '#81c784',
    'Lasso':     '#ff8a65'
}

models_coefs = {
    'LinearReg': model.coef_,
    'Ridge':     ridege_model.coef_,
    'Lasso':     lasso_model.coef_
}

for ax, (model_name, coefs) in zip(axes, models_coefs.items()):
    ax.set_title(f'{model_name}', color='white', fontsize=13, fontweight='bold')
    ax.set_facecolor('#1a1a1a')

    bar_colors = ['#a78bfa' if 'noise' not in name else '#ef4444'
                  for name in feature_names]

    bars = ax.barh(feature_names, coefs, color=bar_colors, edgecolor='white',
                   linewidth=0.4)

    ax.axvline(x=0, color='white', linestyle='--', linewidth=1.2, alpha=0.6)

    ax.set_title(f'{model_name}\nR² = {r2_score(np.exp(y_test), np.exp(locals()[model_name.lower() + "_log_pred"]) if model_name != "LinearReg" else np.exp(model.predict(X_test_scaled))):.4f}',
                 color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('Coefficient Value', color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    for i, (name, coef) in enumerate(zip(feature_names, coefs)):
        if abs(coef) < 1e-6:
            ax.text(0.001, i, '← ZEROED', color='#ff8a65',
                    fontsize=8, va='center')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#a78bfa', label='Real feature'),
    Patch(facecolor='#ef4444', label='Noise feature (fake)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           facecolor='#1a1a1a', edgecolor='white', labelcolor='white',
           fontsize=11, bbox_to_anchor=(0.5, -0.05))

plt.suptitle('Linear vs Ridge vs Lasso — Coefficient Comparison\n(Purple = real features | Red = injected noise)',
             color='white', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()