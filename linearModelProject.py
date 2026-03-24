import numpy as np
import pandas as pd
 

np.random.seed(42)
N = 1000  # number of tablet formulation experiments
 
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

X = df.drop(['dissolution_rate_pct'],axis=0.5)
y = df['dissolution_rate_pct']

print(X.shape,y.shape)


#Linear model selection