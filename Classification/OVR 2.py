import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


# 1. GENERATE RANDOM DATA (3 classes)

X, y = make_classification(
    n_samples=500,
    n_features=2,        # 2 features for easy plotting
    n_informative=2,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

class_names = ['Class A', 'Class B', 'Class C']
colors      = ['#FF6B6B', '#4ECDC4', '#45B7D1']


# 2. SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# 3. TRAIN MODEL

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)


# 4. PLOT EVERYTHING

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Multiclass Classification — Full Visualization', fontsize=16, fontweight='bold', y=1.01)

# ── Plot 1: Raw Data Scatter ──────────────
ax = axes[0, 0]
for i, (name, color) in enumerate(zip(class_names, colors)):
    mask = y == i
    ax.scatter(X[mask, 0], X[mask, 1], c=color, label=name, alpha=0.7, edgecolors='white', s=60)
ax.set_title('Raw Data (3 Classes)', fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.grid(True, alpha=0.3)

# ── Plot 2: Decision Boundary ─────────────
ax = axes[0, 1]
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cmap_bg = ListedColormap(['#FFB3B3', '#B3EFEA', '#B3DFF5'])
ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
for i, (name, color) in enumerate(zip(class_names, colors)):
    mask = y_train == i
    ax.scatter(X_train[mask, 0], X_train[mask, 1], c=color, label=name,
               edgecolors='white', s=50, alpha=0.8)
ax.set_title('Decision Boundary', fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.grid(True, alpha=0.3)

# ── Plot 3: Confusion Matrix ──────────────
ax = axes[0, 2]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white')
ax.set_title(f'Confusion Matrix (Acc: {acc:.2%})', fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# ── Plot 4: Class Distribution ────────────
ax = axes[1, 0]
unique, counts = np.unique(y, return_counts=True)
bars = ax.bar(class_names, counts, color=colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(count), ha='center', fontweight='bold')
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Count')
ax.set_ylim(0, max(counts) + 20)
ax.grid(True, alpha=0.3, axis='y')

# ── Plot 5: Prediction Probability Bars ──
ax = axes[1, 1]
probs     = model.predict_proba(X_test[:10])   # first 10 test samples
x_indices = np.arange(10)
width     = 0.25
for i, (name, color) in enumerate(zip(class_names, colors)):
    ax.bar(x_indices + i * width, probs[:, i], width, label=name, color=color, alpha=0.85)
ax.set_title('Prediction Probabilities (First 10 Samples)', fontweight='bold')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Probability')
ax.set_xticks(x_indices + width)
ax.set_xticklabels([f'S{i}' for i in range(10)])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# ── Plot 6: Precision / Recall / F1 ──────
ax = axes[1, 2]
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
x       = np.arange(len(class_names))
width   = 0.25
metric_colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']
for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
    vals = [report[cls][metric] for cls in class_names]
    ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color, alpha=0.85)
ax.set_title('Precision / Recall / F1 per Class', fontweight='bold')
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_xticks(x + width)
ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('multiclass_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n Final Accuracy: {acc:.2%}")