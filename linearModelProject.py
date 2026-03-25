import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score






np.random.seed(42)
N = 700  # number of tablet formulation experiments
 
binder_concentration  = np.random.uniform(2, 15, N)          # %
disintegrant_ratio    = np.random.uniform(0.01, 0.15, N)     # ratio
compression_force     = np.random.uniform(5, 25, N)          # kN
particle_size_um      = np.random.uniform(50, 500, N)        # microns
drug_load_pct         = np.random.uniform(10, 60, N)         # %
hydrophilicity_index  = np.random.uniform(-2, 5, N)          # log P
tablet_porosity       = 35 - 0.9 * compression_force \
                        + np.random.normal(0, 2, N)           # % (correlated with compression!)
tablet_porosity       = np.clip(tablet_porosity, 5, 30)
lubricant_pct         = np.random.uniform(0.1, 2, N)         # %
moisture_content      = np.random.uniform(1, 6, N)           # %
coating_thickness_um  = np.random.uniform(0, 150, N)         # microns
 
noise_feature_1 = binder_concentration + np.random.normal(0, 3, N)   # near-duplicate
noise_feature_2 = np.random.uniform(0, 1, N)                          # pure noise
 
 
dissolution_rate = (
      85
    + 3.5  * disintegrant_ratio * 100       # strong positive
    + 0.9  * tablet_porosity                # positive
    - 1.2  * compression_force              # negative
    - 0.05 * particle_size_um               # negative
    + 2.0  * binder_concentration           # mild positive
    - 0.8  * hydrophilicity_index           # lower logP = more hydrophilic = faster
    - 0.15 * coating_thickness_um           # coating slows release
    - 4.5  * lubricant_pct                  # hydrophobic barrier
    - 0.3  * drug_load_pct                  # dilution effect
    + 0.8  * moisture_content               # slight swelling aids disintegration
    - 0.002 * compression_force * particle_size_um  # interaction term
    + np.random.normal(0, 4, N)             # lab measurement noise (±4% realistic)
)
 
dissolution_rate = np.clip(dissolution_rate, 5, 100)
 
outlier_idx = np.random.choice(N, size=int(0.03 * N), replace=False)
dissolution_rate[outlier_idx] += np.random.choice([-30, 30], size=len(outlier_idx))
dissolution_rate = np.clip(dissolution_rate, 5, 100)
 
df = pd.DataFrame({
    "binder_concentration":  binder_concentration,
    "disintegrant_ratio":    disintegrant_ratio,
    "compression_force":     compression_force,
    "particle_size_um":      particle_size_um,
    "drug_load_pct":         drug_load_pct,
    "hydrophilicity_index":  hydrophilicity_index,
    "tablet_porosity":       tablet_porosity,
    "lubricant_pct":         lubricant_pct,
    "moisture_content":      moisture_content,
    "coating_thickness_um":  coating_thickness_um,
    "noise_feature_1":       noise_feature_1,   # redundant — Lasso should zero this
    "noise_feature_2":       noise_feature_2,   # pure noise — should be zeroed
    "dissolution_rate_pct":  dissolution_rate,   # TARGET
})
 
# df.to_csv("./drug_dissolution_dataset1.csv", index=False)
 
# print("=" * 60)
# print("  DRUG DISSOLUTION SYNTHETIC DATASET — SUMMARY")
# print("=" * 60)
# print(f"\n  Samples : {N}")
# print(f"  Features: {df.shape[1] - 1}")
# print(f"  Target  : dissolution_rate_pct\n")
# print(df.describe().T[["mean", "std", "min", "max"]].round(2))
# print("\n  Correlation with target (top features):")
# corr = df.corr()["dissolution_rate_pct"].drop("dissolution_rate_pct").sort_values()
# print(corr.round(3).to_string())
# print("\n  Files saved:")
# print("    drug_dissolution_dataset1.csv")
# print("=" * 60)

#features selection

X = df.drop(['dissolution_rate_pct'],axis=1)
y = df['dissolution_rate_pct']

print(X.shape,y.shape)



#Linear model Train_test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection
LinearModel = LinearRegression()
LinearModel.fit(X_train,y_train)


print(f"Features model expects: {LinearModel.feature_names_in_}")
print(f"Number of features: {LinearModel.n_features_in_}")
#model predicting and evaluating
y_linear_predict = LinearModel.predict(X_test)


y_new_value = [[
    14.608828078105926, 0.14827352111953057, 6.25306406910709,
    179.4347280696552, 29.595528147195452, 2.9462289029595246,
    30.0, 1.0577331515220063, 5.23648372767747, 109.84511559122716,
    15.62575575911071, 0.6899690121417249,
]]

result_new_value = LinearModel.predict(y_new_value)

print(f"new value prediction: {result_new_value}")

#evaluation

print(f"mse of linear model is {mean_squared_error(y_test,y_linear_predict)}")
print(f"r2_score of linear model is {r2_score(y_test,y_linear_predict)}")


plt.figure(figsize=(10,8))
plt.scatter(y_test,y_linear_predict,label='Actual data vs predicted')
plt.plot([y_test.min(),y_test.max()],[y_test.min(), y_test.max()], lw=2, label='Ideal fit (y=x)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(10,8))
plt.scatter(y_test,y_linear_predict,label='Actual vs predicted data')
plt.plot(y_new_value,result_new_value,label='New value prediction',marker='*')
plt.xlabel('Actual values')
plt.ylabel('predicted values')
plt.title('New value prediction are')
plt.grid(True)
plt.show()


#Polynomial linear regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)


poly_model = LinearRegression()
poly_model.fit(X_train,y_train)

#predict and evaluate
y_poly_pred = poly_model.predict(X_test)


print("Polynomial Regression Results")
print("R2 Score:", r2_score(y_test, y_poly_pred))
print("MSE:", mean_squared_error(y_test, y_poly_pred))


np.random.seed(42)
N = 1000
 
binder_concentration = np.random.uniform(2, 15, N)
disintegrant_ratio   = np.random.uniform(0.01, 0.15, N)
compression_force    = np.random.uniform(5, 25, N)
particle_size_um     = np.random.uniform(50, 500, N)
drug_load_pct        = np.random.uniform(10, 60, N)
hydrophilicity_index = np.random.uniform(-2, 5, N)
tablet_porosity      = np.clip(35 - 0.9 * compression_force + np.random.normal(0, 2, N), 5, 30)
lubricant_pct        = np.random.uniform(0.1, 2, N)
moisture_content     = np.random.uniform(1, 6, N)
coating_thickness_um = np.random.uniform(0, 150, N)
noise_feature_1      = binder_concentration + np.random.normal(0, 3, N)
noise_feature_2      = np.random.uniform(0, 1, N)
 
dissolution_rate = np.clip(
      85
    + 3.5  * disintegrant_ratio * 100
    + 0.9  * tablet_porosity
    - 1.2  * compression_force
    - 0.05 * particle_size_um
    + 2.0  * binder_concentration
    - 0.8  * hydrophilicity_index
    - 0.15 * coating_thickness_um
    - 4.5  * lubricant_pct
    - 0.3  * drug_load_pct
    + 0.8  * moisture_content
    - 0.002 * compression_force * particle_size_um
    + np.random.normal(0, 4, N),
    5, 100
)
outlier_idx = np.random.choice(N, size=int(0.03 * N), replace=False)
dissolution_rate[outlier_idx] += np.random.choice([-30, 30], size=len(outlier_idx))
dissolution_rate = np.clip(dissolution_rate, 5, 100)
 
df = pd.DataFrame({
    "binder_concentration": binder_concentration,
    "disintegrant_ratio":   disintegrant_ratio,
    "compression_force":    compression_force,
    "particle_size_um":     particle_size_um,
    "drug_load_pct":        drug_load_pct,
    "hydrophilicity_index": hydrophilicity_index,
    "tablet_porosity":      tablet_porosity,
    "lubricant_pct":        lubricant_pct,
    "moisture_content":     moisture_content,
    "coating_thickness_um": coating_thickness_um,
    "noise_feature_1":      noise_feature_1,
    "noise_feature_2":      noise_feature_2,
    "dissolution_rate_pct": dissolution_rate,
})
 
X = df.drop("dissolution_rate_pct", axis=1)
y = df["dissolution_rate_pct"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 

results = {}
degrees = [1, 2, 3]
colors  = ["#4C9BE8", "#E8834C", "#4CE87A"]
 
for deg in degrees:
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results[deg] = {
        "pipeline": pipe,
        "y_pred":   y_pred,
        "r2":       r2_score(y_test, y_pred),
        "rmse":     np.sqrt(mean_squared_error(y_test, y_pred)),
        "residuals": y_test.values - y_pred,
    }
    print(f"Degree {deg} | R² = {results[deg]['r2']:.4f} | RMSE = {results[deg]['rmse']:.4f}")

new_sample = pd.DataFrame([{
    "binder_concentration":  14.608828078105926,
    "disintegrant_ratio":    0.14827352111953057,
    "compression_force":     6.25306406910709,
    "particle_size_um":      179.4347280696552,
    "drug_load_pct":         29.595528147195452,
    "hydrophilicity_index":  2.9462289029595246,
    "tablet_porosity":       30.0,
    "lubricant_pct":         1.0577331515220063,
    "moisture_content":      5.23648372767747,
    "coating_thickness_um":  109.84511559122716,
    "noise_feature_1":       15.62575575911071,
    "noise_feature_2":       0.6899690121417249,
}])
 
print("\n── New Sample Predictions ──")
for deg in degrees:
    pred = results[deg]["pipeline"].predict(new_sample)[0]
    print(f"  Degree {deg} → {pred:.2f}% dissolution")
 
# # ─────────────────────────────────────────────
# #  PLOTTING  — 6-panel dashboard
# # ─────────────────────────────────────────────
# plt.style.use("dark_background")
# DARK   = "#0E1117"
# PANEL  = "#1A1D27"
# ACCENT = "#A78BFA"
# TEXT   = "#E2E8F0"
 
# fig = plt.figure(figsize=(20, 16), facecolor=DARK)
# fig.suptitle(
#     "Polynomial Regression — Drug Dissolution Rate Prediction",
#     fontsize=18, fontweight="bold", color=TEXT, y=0.98
# )
 
# gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
 
# # ── Panel 1–3 : Actual vs Predicted per degree ──────────────────
# for i, (deg, col) in enumerate(zip(degrees, colors)):
#     ax = fig.add_subplot(gs[0, i])
#     ax.set_facecolor(PANEL)
#     y_pred = results[deg]["y_pred"]
 
#     ax.scatter(y_test, y_pred, alpha=0.45, s=18, color=col, edgecolors="none")
#     mn, mx = y_test.min(), y_test.max()
#     ax.plot([mn, mx], [mn, mx], color="white", lw=1.5, ls="--", label="Ideal fit")
 
#     ax.set_title(f"Degree {deg}  |  R²={results[deg]['r2']:.3f}",
#                  fontsize=11, color=TEXT, pad=8)
#     ax.set_xlabel("Actual (%)",  color=TEXT, fontsize=9)
#     ax.set_ylabel("Predicted (%)", color=TEXT, fontsize=9)
#     ax.tick_params(colors=TEXT, labelsize=8)
#     for spine in ax.spines.values():
#         spine.set_edgecolor("#2D3148")
#     ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
 
# # ── Panel 4–6 : Residual distributions per degree ───────────────
# for i, (deg, col) in enumerate(zip(degrees, colors)):
#     ax = fig.add_subplot(gs[1, i])
#     ax.set_facecolor(PANEL)
#     residuals = results[deg]["residuals"]
 
#     ax.hist(residuals, bins=35, color=col, alpha=0.75, edgecolor="none")
#     ax.axvline(0, color="white", lw=1.5, ls="--")
#     ax.axvline(residuals.mean(), color=ACCENT, lw=1.5, ls="-",
#                label=f"Mean={residuals.mean():.2f}")
 
#     ax.set_title(f"Residuals — Degree {deg}", fontsize=11, color=TEXT, pad=8)
#     ax.set_xlabel("Residual (%)", color=TEXT, fontsize=9)
#     ax.set_ylabel("Count",        color=TEXT, fontsize=9)
#     ax.tick_params(colors=TEXT, labelsize=8)
#     for spine in ax.spines.values():
#         spine.set_edgecolor("#2D3148")
#     ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
 
# # ── Panel 7 : R² bar chart comparison ───────────────────────────
# ax7 = fig.add_subplot(gs[2, 0])
# ax7.set_facecolor(PANEL)
# r2_vals  = [results[d]["r2"]   for d in degrees]
# rmse_vals= [results[d]["rmse"] for d in degrees]
# bars = ax7.bar([f"Degree {d}" for d in degrees], r2_vals,
#                color=colors, alpha=0.85, edgecolor="none", width=0.5)
# for bar, val in zip(bars, r2_vals):
#     ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
#              f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
# ax7.set_ylim(0, 1.08)
# ax7.set_title("R² Score Comparison", fontsize=11, color=TEXT, pad=8)
# ax7.set_ylabel("R²", color=TEXT, fontsize=9)
# ax7.tick_params(colors=TEXT, labelsize=9)
# for spine in ax7.spines.values():
#     spine.set_edgecolor("#2D3148")
 
# # ── Panel 8 : RMSE bar chart comparison ─────────────────────────
# ax8 = fig.add_subplot(gs[2, 1])
# ax8.set_facecolor(PANEL)
# bars2 = ax8.bar([f"Degree {d}" for d in degrees], rmse_vals,
#                 color=colors, alpha=0.85, edgecolor="none", width=0.5)
# for bar, val in zip(bars2, rmse_vals):
#     ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
#              f"{val:.2f}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
# ax8.set_title("RMSE Comparison", fontsize=11, color=TEXT, pad=8)
# ax8.set_ylabel("RMSE (%)", color=TEXT, fontsize=9)
# ax8.tick_params(colors=TEXT, labelsize=9)
# for spine in ax8.spines.values():
#     spine.set_edgecolor("#2D3148")
 
# # ── Panel 9 : New sample prediction gauge ───────────────────────
# ax9 = fig.add_subplot(gs[2, 2])
# ax9.set_facecolor(PANEL)
# preds = [results[d]["pipeline"].predict(new_sample)[0] for d in degrees]
# bars3 = ax9.barh([f"Degree {d}" for d in degrees], preds,
#                  color=colors, alpha=0.85, edgecolor="none", height=0.45)
# for bar, val in zip(bars3, preds):
#     ax9.text(val + 0.5, bar.get_y() + bar.get_height()/2,
#              f"{val:.1f}%", va="center", color=TEXT, fontsize=10, fontweight="bold")
# ax9.set_xlim(0, 115)
# ax9.axvline(80, color=ACCENT, lw=1.2, ls="--", alpha=0.6, label="FDA Q30 target (80%)")
# ax9.set_title("New Sample Prediction", fontsize=11, color=TEXT, pad=8)
# ax9.set_xlabel("Predicted Dissolution (%)", color=TEXT, fontsize=9)
# ax9.tick_params(colors=TEXT, labelsize=9)
# for spine in ax9.spines.values():
#     spine.set_edgecolor("#2D3148")
# ax9.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT)
 
# # plt.savefig("/mnt/user-data/outputs/polynomial_regression_dashboard.png",
# #             dpi=150, bbox_inches="tight", facecolor=DARK)
# plt.show()
# print("\nPlot saved → polynomial_regression_dashboard.png")
 

 #ridge regression
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

RidgeModel = Ridge(alpha=0.2)
RidgeModel.fit(X_train_s, y_train)          # ← fit on SCALED data

y_ridge_predict = RidgeModel.predict(X_test_s)  # ← predict on SCALED data

train_r2 = r2_score(y_train, RidgeModel.predict(X_train_s))
test_r2  = r2_score(y_test,  y_ridge_predict)
gap      = train_r2 - test_r2

print(f"Train R²  : {train_r2:.4f}")
print(f"Test  R²  : {test_r2:.4f}")
print(f"Gap       : {gap:.4f}")

if train_r2 < 0.7 and test_r2 < 0.7:
    print("UNDERFITTING — both scores low, model too simple")
elif gap > 0.1:
    print("OVERFITTING  — train >> test, model memorised training data")
else:
    print("GOOD FIT     — train ≈ test, model generalises well")

plt.figure(figsize=(10,8))
plt.plot(y_ridge_predict,color='red',marker='o',label='Actual vs predicted')
plt.show()


cv_scores = cross_val_score(RidgeModel, X_train_s, y_train, cv=5, scoring='r2')

print(f"CV R² scores : {cv_scores.round(4)}")
print(f"CV Mean R²   : {cv_scores.mean():.4f}")
print(f"CV Std R²    : {cv_scores.std():.4f}")
print()

if cv_scores.mean() < 0.7:
    print("UNDERFITTING")
elif cv_scores.std() > 0.1:
    print("OVERFITTING — high variance across folds")
else:
    print("GOOD FIT")


from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=[0.01, 0.1, 0.2, 1.0, 5.0, 10.0], cv=5)
ridge_cv.fit(X_train_s, y_train)
print(f"Best alpha : {ridge_cv.alpha_}")
print(f"Test R²    : {r2_score(y_test, ridge_cv.predict(X_test_s)):.4f}")