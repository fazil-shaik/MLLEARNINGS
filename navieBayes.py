import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 300  # 100 samples per class

pure = pd.DataFrame({
    'capsaicin_index':     np.random.normal(82, 5, n//3),   # high capsaicin
    'red_hue_deviation':   np.random.normal(4, 1.5, n//3),  # low hue shift
    'moisture_pct':        np.random.normal(9, 1, n//3),    # natural moisture
    'particle_uniformity': np.random.normal(88, 4, n//3),   # uniform grind
    'volatile_oil_pct':    np.random.normal(14, 1.5, n//3), # rich oils
    'label': 0  # pure
})

mild = pd.DataFrame({
    'capsaicin_index':     np.random.normal(60, 6, n//3),   # diluted capsaicin
    'red_hue_deviation':   np.random.normal(10, 2, n//3),   # moderate hue shift
    'moisture_pct':        np.random.normal(13, 1.5, n//3), # starch raises moisture
    'particle_uniformity': np.random.normal(70, 6, n//3),   # less uniform
    'volatile_oil_pct':    np.random.normal(9, 1.5, n//3),  # oils diluted
    'label': 1  # mild
})

heavy = pd.DataFrame({
    'capsaicin_index':     np.random.normal(35, 7, n//3),   # very low — mostly filler
    'red_hue_deviation':   np.random.normal(22, 3, n//3),   # strong dye signature
    'moisture_pct':        np.random.normal(18, 2, n//3),   # high moisture (starch)
    'particle_uniformity': np.random.normal(50, 8, n//3),   # irregular (brick grit)
    'volatile_oil_pct':    np.random.normal(4, 1, n//3),    # almost no natural oils
    'label': 2  # heavy
})

df = pd.concat([pure, mild, heavy], ignore_index=True).sample(frac=1, random_state=42)
df['label'] = df['label'].map({0: 'pure', 1: 'mild_adult', 2: 'heavy_adult'})

print(df.head(10).to_string(index=False))
print(f"\nShape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

plt.style.use('dark_background')

g = sns.pairplot(
    df, hue='label',
    vars=['capsaicin_index', 'red_hue_deviation', 'moisture_pct',
          'particle_uniformity', 'volatile_oil_pct'],
    palette={'pure': '#00e676', 'mild_adult': '#ffd740', 'heavy_adult': '#ff5252'},
    plot_kws={'alpha': 0.5, 's': 20},
    diag_kind='kde'
)
g.figure.suptitle('Chilli Powder Adulteration — Feature Space', y=1.02,
                   color='white', fontsize=13)
plt.tight_layout()
plt.savefig('chilli_adulteration.png', dpi=150, bbox_inches='tight',
            facecolor='#121212')
plt.show()