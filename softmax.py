import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_classification   # synthetic data generator
from sklearn.linear_model import LogisticRegression # softmax model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA              # for 2D visualization
import seaborn as sns


# STEP 1 — GENERATE SYNTHETIC DATA


X, y = make_classification(
    n_samples=600,          # total 600 data points
    n_features=10,          # 10 input features per sample
    n_informative=5,        # only 5 features actually carry signal
    n_redundant=2,          # 2 features are noisy combos of the 5 above
    n_classes=3,            # 3 output classes → softmax will output 3 probs
    n_clusters_per_class=1, # each class forms one blob in feature space
    random_state=42         # fix seed for reproducibility
)



class_names = ['Class A', 'Class B', 'Class C']

print("X shape:", X.shape)               # (600, 10)
print("y distribution:", np.bincount(y)) # ~200 samples per class


# STEP 2 — TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,    # 30% for testing (180 samples), 70% for training (420)
    stratify=y,       # ensures each class is equally represented in both splits
    random_state=42
)


# STEP 3 — FEATURE SCALING


scaler = StandardScaler()
# fit_transform on train: computes mean & std from training data, then scales it
X_train_sc = scaler.fit_transform(X_train)

# transform on test: uses the SAME mean & std from training (no data leakage!)
X_test_sc  = scaler.transform(X_test)

# STEP 4 — TRAIN THE SOFTMAX REGRESSION MODEL


model = LogisticRegression(
    solver='lbfgs',   # optimization algorithm; supports softmax natively
    max_iter=500,     # max gradient descent steps before stopping
    random_state=42
)
model.fit(X_train_sc, y_train)


# STEP 5 — MAKE PREDICTIONS
y_pred  = model.predict(X_test_sc)


y_proba = model.predict_proba(X_test_sc)


acc = (y_pred == y_test).mean()
print(f"Test Accuracy: {acc:.1%}")


# STEP 6 — EVALUATION REPORT


print(classification_report(y_test, y_pred, target_names=class_names))


# STEP 7 — PRINT SOFTMAX PROBABILITIES FOR 5 SAMPLES


print("\nSample softmax output (first 5 test points):")
print(f"{'True':>10} {'Pred':>10}  {'P(A)':>8} {'P(B)':>8} {'P(C)':>8}")
for i in range(5):
    print(f"{class_names[y_test[i]]:>10} {class_names[y_pred[i]]:>10}  "
          f"{y_proba[i,0]:>8.3f} {y_proba[i,1]:>8.3f} {y_proba[i,2]:>8.3f}")
# Notice: probabilities always sum to 1.0 — that is the softmax guarantee


# STEP 8 — VISUALIZATIONS

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('#0f0f17')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
palette = ['#5DD6F0', '#F0C040', '#F06878']


ax0 = fig.add_subplot(gs[0, 0])
cm  = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax0, cbar=False,
            annot_kws={'fontsize':14, 'color':'white'})
ax0.set_facecolor('#1a1a2e')
ax0.set_title('Confusion Matrix\n(rows=actual, cols=predicted)',
              color='white', fontsize=11, pad=8)
ax0.set_xlabel('Predicted', color='#aaa', fontsize=10)
ax0.set_ylabel('Actual',    color='#aaa', fontsize=10)
ax0.tick_params(colors='#aaa', labelsize=9)


ax1 = fig.add_subplot(gs[0, 1:])
n_show   = 20
probs_20 = y_proba[:n_show]
x_pos    = np.arange(n_show)
bottoms  = np.zeros(n_show)

for i, (cls, col) in enumerate(zip(class_names, palette)):
    ax1.bar(x_pos, probs_20[:, i], bottom=bottoms,
            color=col, label=cls, alpha=0.88, edgecolor='none')
    bottoms += probs_20[:, i]  # stack next class on top

for i in range(n_show):
    if y_pred[i] != y_test[i]:
        ax1.text(i, 1.03, '✗', ha='center', va='bottom',
                 color='#ff4444', fontsize=11, fontweight='bold')

ax1.set_facecolor('#1a1a2e')
ax1.set_title('Stacked Softmax Probabilities — first 20 test samples\n'
              '(each bar sums to 1.0  ·  ✗ = misclassified)',
              color='white', fontsize=11, pad=8)
ax1.set_xlabel('Sample index', color='#aaa', fontsize=10)
ax1.set_ylabel('Probability',  color='#aaa', fontsize=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_pos, fontsize=7, color='#aaa')
ax1.tick_params(axis='y', colors='#aaa', labelsize=9)
ax1.legend(loc='lower right', fontsize=9,
           facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
ax1.spines[['top','right']].set_visible(False)
ax1.spines[['left','bottom']].set_color('#333')


ax2 = fig.add_subplot(gs[1, :2])
pca    = PCA(n_components=2, random_state=42)
X_tr2  = pca.fit_transform(X_train_sc)  # 420 samples, now 2D
X_te2  = pca.transform(X_test_sc)

model2 = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42)
model2.fit(X_tr2, y_train)   # retrain on 2D data for boundary plotting

# Create a fine mesh grid covering the 2D space
h = 0.05
x_min, x_max = X_tr2[:,0].min()-0.6, X_tr2[:,0].max()+0.6
y_min, y_max = X_tr2[:,1].min()-0.6, X_tr2[:,1].max()+0.6
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict the class at every grid point → fills the background color
Z = model2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax2.contourf(xx, yy, Z, alpha=0.30,
             levels=[-0.5,0.5,1.5,2.5],
             colors=['#1a2a3a','#2a2a1a','#2a1a1a'])
ax2.contour(xx, yy, Z, levels=[0.5,1.5],  # the actual boundary lines
            colors='#666', linewidths=1.0)

for i, (cls, col) in enumerate(zip(class_names, palette)):
    m = y_train == i
    ax2.scatter(X_tr2[m,0], X_tr2[m,1], color=col,
                label=cls, s=25, alpha=0.75, edgecolors='none')

ax2.set_facecolor('#1a1a2e')
ax2.set_title('Linear Decision Boundaries\n(10 features → 2D via PCA for visualization)',
              color='white', fontsize=11, pad=8)
ax2.set_xlabel('PCA Component 1', color='#aaa', fontsize=10)
ax2.set_ylabel('PCA Component 2', color='#aaa', fontsize=10)
ax2.tick_params(colors='#aaa', labelsize=9)
ax2.legend(loc='upper right', fontsize=8,
           facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
ax2.spines[['top','right','left','bottom']].set_color('#333')


ax3 = fig.add_subplot(gs[1, 2])
max_conf = y_proba.max(axis=1)  # pick the highest prob among the 3 classes
correct  = y_pred == y_test

ax3.hist(max_conf[correct],  bins=15, color='#5DD6F0',
         alpha=0.75, label=f'Correct ({correct.sum()})', edgecolor='none')
ax3.hist(max_conf[~correct], bins=15, color='#F06878',
         alpha=0.75, label=f'Wrong ({(~correct).sum()})', edgecolor='none')
ax3.axvline(0.5, color='#888', linestyle='--', linewidth=0.8)

ax3.set_facecolor('#1a1a2e')
ax3.set_title('Model Confidence\n(max softmax probability per sample)',
              color='white', fontsize=11, pad=8)
ax3.set_xlabel('Confidence score', color='#aaa', fontsize=10)
ax3.set_ylabel('# of samples',     color='#aaa', fontsize=10)
ax3.tick_params(colors='#aaa', labelsize=9)
ax3.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
ax3.spines[['top','right']].set_visible(False)
ax3.spines[['left','bottom']].set_color('#333')

fig.suptitle(
    f'Softmax Regression — Synthetic 3-Class Data  |  Test Accuracy: {acc:.1%}',
    color='white', fontsize=15, y=0.98
)

plt.tight_layout()
plt.savefig('softmax_synthetic_explained.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()