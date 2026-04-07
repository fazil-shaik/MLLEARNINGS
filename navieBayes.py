import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.metrics import r2_score,mean_squared_error

#without noise based data exact result with acc of 1.00
# np.random.seed(42)
# n = 300  # 100 samples per class

# pure = pd.DataFrame({
#     'capsaicin_index':     np.random.normal(72, 5, n//3),   # high capsaicin
#     'red_hue_deviation':   np.random.normal(4, 1.5, n//3),  # low hue shift
#     'moisture_pct':        np.random.normal(9, 1, n//3),    # natural moisture
#     'particle_uniformity': np.random.normal(88, 4, n//3),   # uniform grind
#     'volatile_oil_pct':    np.random.normal(14, 1.5, n//3), # rich oils
#     'label': 0  # pure
# })

# mild = pd.DataFrame({
#     'capsaicin_index':     np.random.normal(50, 6, n//3),   # diluted capsaicin
#     'red_hue_deviation':   np.random.normal(10, 2, n//3),   # moderate hue shift
#     'moisture_pct':        np.random.normal(13, 1.5, n//3), # starch raises moisture
#     'particle_uniformity': np.random.normal(70, 6, n//3),   # less uniform
#     'volatile_oil_pct':    np.random.normal(9, 1.5, n//3),  # oils diluted
#     'label': 1  # mild
# })

# heavy = pd.DataFrame({
#     'capsaicin_index':     np.random.normal(65, 7, n//3),   # very low — mostly filler
#     'red_hue_deviation':   np.random.normal(22, 3, n//3),   # strong dye signature
#     'moisture_pct':        np.random.normal(18, 2, n//3),   # high moisture (starch)
#     'particle_uniformity': np.random.normal(50, 8, n//3),   # irregular (brick grit)
#     'volatile_oil_pct':    np.random.normal(4, 1, n//3),    # almost no natural oils
#     'label': 2  # heavy
# })


#noise based data

np.random.seed(42)
n = 300

pure = pd.DataFrame({
    'capsaicin_index':     np.random.normal(72, 8, n//3),   # was 82
    'red_hue_deviation':   np.random.normal(7, 3, n//3),    # was 4
    'moisture_pct':        np.random.normal(10, 2, n//3),   # was 9
    'particle_uniformity': np.random.normal(78, 7, n//3),   # was 88
    'volatile_oil_pct':    np.random.normal(12, 3, n//3),   # was 14
    'label': 0
})

mild = pd.DataFrame({
    'capsaicin_index':     np.random.normal(62, 8, n//3),   # was 60
    'red_hue_deviation':   np.random.normal(12, 3, n//3),   # was 10
    'moisture_pct':        np.random.normal(13, 2, n//3),   # was 13
    'particle_uniformity': np.random.normal(65, 7, n//3),   # was 70
    'volatile_oil_pct':    np.random.normal(8, 3, n//3),    # was 9
    'label': 1
})

heavy = pd.DataFrame({
    'capsaicin_index':     np.random.normal(52, 8, n//3),   # was 35
    'red_hue_deviation':   np.random.normal(17, 3, n//3),   # was 22
    'moisture_pct':        np.random.normal(16, 2, n//3),   # was 18
    'particle_uniformity': np.random.normal(52, 7, n//3),   # was 50
    'volatile_oil_pct':    np.random.normal(5, 3, n//3),    # was 4
    'label': 2
})

df = pd.concat([pure, mild, heavy], ignore_index=True).sample(frac=1, random_state=42)
df['label'] = df['label'].map({0: 'pure', 1: 'mild_adult', 2: 'heavy_adult'})

print(df.head(10).to_string(index=False))
print(f"\nShape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

# plt.style.use('dark_background')

# g = sns.pairplot(
#     df, hue='label',
#     vars=['capsaicin_index', 'red_hue_deviation', 'moisture_pct',
#           'particle_uniformity', 'volatile_oil_pct'],
#     palette={'pure': '#00e676', 'mild_adult': '#ffd740', 'heavy_adult': '#ff5252'},
#     plot_kws={'alpha': 0.5, 's': 20},
#     diag_kind='kde'
# )
# g.figure.suptitle('Chilli Powder Adulteration — Feature Space', y=1.02,
#                    color='white', fontsize=13)
# plt.tight_layout()
# plt.savefig('chilli_adulteration.png', dpi=150, bbox_inches='tight',
#             facecolor='#121212')
# plt.show()

X = df.drop(columns='label')
y = df['label']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

ModelNavies = GaussianNB(var_smoothing=1e-9)
ModelNavies.fit(X_train,y_train)

y_predict = ModelNavies.predict(X_test)

# y_test_predict = ModelNavies.predict(X_test)
# y_train_predict = ModelNavies.predict(X_train)

# # print(f"mse of train data {mean_squared_error(y_train,y_train_predict)}")
# # print(f"mse of test data {mean_squared_error(y_test,y_test)}")
train_acc = ModelNavies.score(X_train, y_train)
test_acc  = ModelNavies.score(X_test, y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test  accuracy: {test_acc:.4f}")

print(f"=========================")
print(f"\nMain report of Navie bayes")
print(f"=========================")
print(f"classification report {classification_report(y_test,y_predict)}")
print(f"accuracy score is {accuracy_score(y_test,y_predict)}")
print(f"confustion matrix {confusion_matrix(y_test,y_predict)}")

confusion = confusion_matrix(y_test,y_predict)

# sns.heatmap(confusion,linecolor='yellow',linewidths=2)
# plt.show()

sns.heatmap(confusion,
            linecolor='yellow',
            linewidths=2,
            annot=True,          # show counts inside cells
            fmt='d',             # integer format
            xticklabels=['pure', 'mild_adult', 'heavy_adult'],
            yticklabels=['pure', 'mild_adult', 'heavy_adult'],
            cmap='YlOrRd')       # dark-friendly colormap

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix — Chilli Adulteration')
plt.show()


cross_score = cross_val_score(cv=5,scoring='accuracy',X=X,y=y,estimator=ModelNavies)
print(f"Mean: {cross_score.mean():.4f}")
print(f"Std:  {cross_score.std():.4f}")
print(cross_score)

