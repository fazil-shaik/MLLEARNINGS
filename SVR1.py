import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# 1. LOAD & PREPARE DATA

data = fetch_california_housing(as_frame=True)
df = data.frame

print("Dataset shape:", df.shape)
print("\nFeatures:", data.feature_names)
print("\nTarget: Median house value (in $100k)")
print(df.describe())

# Use a subset for speed (SVR is O(n²) in memory)
df_sample = df.sample(n=2000, random_state=42)

X = df_sample[data.feature_names]
y = df_sample['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")



# 2. WHY SCALING IS CRITICAL FOR SVR

# SVR uses dot products / distances. If one feature is in thousands
# and another in 0-1, the larger feature dominates. ALWAYS scale.
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()



# 3. COMPARE ALL KERNELS

kernels = {
    'Linear':     SVR(kernel='linear',  C=1.0, epsilon=0.1),
    'Polynomial': SVR(kernel='poly',    C=1.0, epsilon=0.1, degree=3, gamma='scale'),
    'RBF':        SVR(kernel='rbf',     C=1.0, epsilon=0.1, gamma='scale'),
    'Sigmoid':    SVR(kernel='sigmoid', C=1.0, epsilon=0.1, gamma='scale'),
}

results = {}
for name, model in kernels.items():
    model.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Inverse transform predictions back to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    n_sv = model.n_support_[0]  # Number of support vectors
    
    results[name] = {'R²': r2, 'MSE': mse, 'MAE': mae, 'Support Vectors': n_sv, 'model': model, 'pred': y_pred}
    print(f"{name:12s} | R²={r2:.4f} | MSE={mse:.4f} | MAE={mae:.4f} | SVs={n_sv}")



# 4. HYPERPARAMETER TUNING (RBF Kernel)

print("\n--- Grid Search for best RBF SVR ---")

param_grid = {
    'C':       [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma':   ['scale', 'auto', 0.01, 0.1],
}

grid_search = GridSearchCV(
    SVR(kernel='rbf'),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_scaled, y_train_scaled)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV R²:  {grid_search.best_score_:.4f}")

best_svr = grid_search.best_estimator_
y_pred_best_scaled = best_svr.predict(X_test_scaled)
y_pred_best = scaler_y.inverse_transform(y_pred_best_scaled.reshape(-1, 1)).ravel()

print(f"Tuned RBF SVR Test R²:  {r2_score(y_test, y_pred_best):.4f}")
print(f"Tuned RBF SVR Test MAE: {mean_absolute_error(y_test, y_pred_best):.4f}")
print(f"Number of Support Vectors: {best_svr.n_support_[0]}")



# 5. VISUALIZE EPSILON TUBE EFFECT (1D Demo)

# Create a clean 1D dataset to show the tube visually
np.random.seed(0)
X_1d = np.sort(5 * np.random.rand(80))
y_1d = np.sin(X_1d) + np.random.normal(0, 0.2, 80)

X_1d_r = X_1d.reshape(-1, 1)
X_plot  = np.linspace(0, 5, 300).reshape(-1, 1)

epsilons = [0.05, 0.2, 0.5]
colors   = ['#e63946', '#2a9d8f', '#e9c46a']

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Row 1: Epsilon Tube Effect ──
for idx, (eps, col) in enumerate(zip(epsilons, colors)):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_facecolor('#161b22')
    
    svr = SVR(kernel='rbf', C=1.0, epsilon=eps, gamma='scale')
    svr.fit(X_1d_r, y_1d)
    y_hat = svr.predict(X_plot)
    
    ax.scatter(X_1d, y_1d, color='#8b949e', s=20, alpha=0.7, zorder=2)
    ax.plot(X_plot, y_hat,               color=col, linewidth=2.5, zorder=3, label='SVR fit')
    ax.fill_between(X_plot.ravel(), y_hat - eps, y_hat + eps,
                    color=col, alpha=0.2, label=f'ε={eps} tube')
    
    # Highlight support vectors
    sv_mask = np.abs(y_1d - svr.predict(X_1d_r)) >= eps * 0.99
    ax.scatter(X_1d[sv_mask], y_1d[sv_mask], color=col, s=60,
               edgecolors='white', linewidths=1, zorder=4, label=f'SVs ({sv_mask.sum()})')
    
    ax.set_title(f'ε = {eps}  |  {sv_mask.sum()} support vectors',
                 color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('x', color='#8b949e')
    ax.set_ylabel('y', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values(): spine.set_color('#30363d')
    ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white', framealpha=0.8)


# ── Row 2: C Regularization Effect ──
C_values = [0.1, 1, 100]
for idx, (C_val, col) in enumerate(zip(C_values, colors)):
    ax = fig.add_subplot(gs[1, idx])
    ax.set_facecolor('#161b22')
    
    svr = SVR(kernel='rbf', C=C_val, epsilon=0.1, gamma='scale')
    svr.fit(X_1d_r, y_1d)
    y_hat = svr.predict(X_plot)
    
    ax.scatter(X_1d, y_1d, color='#8b949e', s=20, alpha=0.7, zorder=2)
    ax.plot(X_plot, y_hat, color=col, linewidth=2.5, zorder=3)
    ax.fill_between(X_plot.ravel(), y_hat - 0.1, y_hat + 0.1,
                    color=col, alpha=0.2)
    
    ax.set_title(f'C = {C_val}  ({"tight fit" if C_val > 10 else "smooth" if C_val < 1 else "balanced"})',
                 color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('x', color='#8b949e')
    ax.set_ylabel('y', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values(): spine.set_color('#30363d')


# ── Row 3: Kernel Comparison (actual predictions) ──
ax_pred = fig.add_subplot(gs[2, :2])
ax_pred.set_facecolor('#161b22')

# Predicted vs Actual for tuned RBF
ax_pred.scatter(y_test, y_pred_best, alpha=0.5, color='#2a9d8f', s=15, label='Predicted')
lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
ax_pred.plot(lims, lims, 'r--', linewidth=2, label='Perfect fit')
ax_pred.set_title(f'Tuned RBF SVR: Predicted vs Actual  (R²={r2_score(y_test, y_pred_best):.3f})',
                  color='white', fontsize=11, fontweight='bold')
ax_pred.set_xlabel('Actual Price ($100k)', color='#8b949e')
ax_pred.set_ylabel('Predicted Price ($100k)', color='#8b949e')
ax_pred.tick_params(colors='#8b949e')
for spine in ax_pred.spines.values(): spine.set_color('#30363d')
ax_pred.legend(fontsize=9, facecolor='#21262d', labelcolor='white')


# ── Row 3 Right: Kernel Comparison Bar Chart ──
ax_bar = fig.add_subplot(gs[2, 2])
ax_bar.set_facecolor('#161b22')

kernel_names = list(results.keys())
r2_scores    = [results[k]['R²'] for k in kernel_names]
bar_colors   = ['#e63946', '#f4a261', '#2a9d8f', '#a8dadc']

bars = ax_bar.barh(kernel_names, r2_scores, color=bar_colors, height=0.5)
ax_bar.set_xlim(0, 1)
ax_bar.set_title('Kernel R² Comparison', color='white', fontsize=11, fontweight='bold')
ax_bar.set_xlabel('R² Score', color='#8b949e')
ax_bar.tick_params(colors='#8b949e')
for spine in ax_bar.spines.values(): spine.set_color('#30363d')
for bar, val in zip(bars, r2_scores):
    ax_bar.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='white', fontsize=10, fontweight='bold')


# Title
fig.suptitle('Support Vector Regression — Complete Visual Guide',
             color='white', fontsize=16, fontweight='bold', y=0.98)

plt.savefig('svr_complete.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.show()
print("\nPlot saved as svr_complete.png")



# 6. UNDERSTANDING SUPPORT VECTORS

print("\n─── Support Vector Analysis ───")
print(f"Total training points:  {X_train_scaled.shape[0]}")
print(f"Support vectors (RBF):  {best_svr.n_support_[0]}")
print(f"Support vector ratio:   {best_svr.n_support_[0]/X_train_scaled.shape[0]:.1%}")
print("\nOnly these points define the model. All others are irrelevant.")
print("This is what makes SVR memory-efficient and robust to outliers.")



# 7. PIPELINE APPROACH (Production-Ready)

print("\n─── Production Pipeline ───")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale'))
])

pipeline.fit(X_train, y_train_scaled)  # Note: still need to scale y separately
y_pipe_pred_scaled = pipeline.predict(X_test)
y_pipe_pred = scaler_y.inverse_transform(y_pipe_pred_scaled.reshape(-1,1)).ravel()

print(f"Pipeline R²:  {r2_score(y_test, y_pipe_pred):.4f}")
print(f"Pipeline MAE: {mean_absolute_error(y_test, y_pipe_pred):.4f} ($100k)")
