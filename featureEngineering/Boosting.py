import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic credit card transaction data
# Features: amount, time_of_day, location_score, previous_fraud, velocity, device_score
np.random.seed(42)
X, y = make_classification(
    n_samples=5000,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=1,
    weights=[0.95, 0.05],  # Imbalanced: 95% normal, 5% fraud
    flip_y=0.01,
    random_state=42
)

# Create feature names for better interpretation
feature_names = [
    'transaction_amount',
    'time_of_day', 
    'location_risk_score',
    'previous_fraud_count',
    'transaction_velocity',
    'device_risk_score',
    'merchant_category_risk',
    'ip_risk_score'
]

df = pd.DataFrame(X, columns=feature_names)
df['is_fraud'] = y

print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
print("\nFirst few transactions:")
print(df.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


# Train AdaBoost with different weak learner configurations


# AdaBoost with default parameters (Decision Stump as base estimator)
print("\n" + "="*60)
print("1. ADA BOOST WITH DECISION STUMPS (Default)")
print("="*60)

ada_stump = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Decision stump
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada_stump.fit(X_train, y_train)
y_pred_stump = ada_stump.predict(X_test)
y_pred_proba_stump = ada_stump.predict_proba(X_test)[:, 1]

print(f"Accuracy: {ada_stump.score(X_test, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_stump):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_stump, target_names=['Normal', 'Fraud']))


# AdaBoost with deeper trees (more powerful base learners)

print("\n" + "="*60)
print("2. ADA BOOST WITH WEAK TREES (depth=3)")
print("="*60)

ada_tree = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada_tree.fit(X_train, y_train)
y_pred_tree = ada_tree.predict(X_test)
y_pred_proba_tree = ada_tree.predict_proba(X_test)[:, 1]

print(f"Accuracy: {ada_tree.score(X_test, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_tree):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree, target_names=['Normal', 'Fraud']))


# Compare with single decision tree

print("\n" + "="*60)
print("3. SINGLE DECISION TREE (For comparison)")
print("="*60)

single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
single_tree.fit(X_train, y_train)
y_pred_single = single_tree.predict(X_test)
y_pred_proba_single = single_tree.predict_proba(X_test)[:, 1]

print(f"Accuracy: {single_tree.score(X_test, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_single):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_single, target_names=['Normal', 'Fraud']))


# Visualize AdaBoost learning process

# Track performance as we add more estimators
n_estimators_range = range(1, 101, 5)
train_scores = []
test_scores = []

for n in n_estimators_range:
    ada = AdaBoostClassifier(
        n_estimators=n,
        learning_rate=1.0,
        random_state=42
    )
    ada.fit(X_train, y_train)
    train_scores.append(ada.score(X_train, y_train))
    test_scores.append(ada.score(X_test, y_test))

# Plot learning curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Learning curves
axes[0].plot(n_estimators_range, train_scores, label='Training Accuracy', marker='o')
axes[0].plot(n_estimators_range, test_scores, label='Test Accuracy', marker='s')
axes[0].set_xlabel('Number of Estimators')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('AdaBoost Learning Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Confusion Matrix for best model
cm = confusion_matrix(y_test, y_pred_stump)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
axes[1].set_title('Confusion Matrix - AdaBoost with Stumps')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()


# Feature importance analysis

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': ada_stump.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features for fraud detection:")
print(feature_importance.head(5))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance Score')
plt.title('AdaBoost Feature Importance for Fraud Detection')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Sample prediction with real-time explanation

print("\n" + "="*60)
print("REAL-TIME FRAUD DETECTION EXAMPLE")
print("="*60)

# Create a sample transaction
sample_transaction = np.array([[
    5000,    # high transaction amount
    14,      # 2 PM
    0.9,     # high location risk
    2,       # previous frauds from this account
    5,       # 5 transactions in last hour (high velocity)
    0.85,    # suspicious device
    0.7,     # high risk merchant
    0.6      # suspicious IP
]])

# Make prediction
fraud_probability = ada_stump.predict_proba(sample_transaction)[0][1]
prediction = ada_stump.predict(sample_transaction)[0]

print(f"Transaction Details:")
print(f"  Amount: ${sample_transaction[0][0]:.2f}")
print(f"  Time: {sample_transaction[0][1]}:00")
print(f"  Location Risk: {sample_transaction[0][2]:.2f}")
print(f"  Previous Frauds: {sample_transaction[0][3]}")
print(f"  Transaction Velocity: {sample_transaction[0][4]} transactions/hour")
print(f"  Device Risk: {sample_transaction[0][5]:.2f}")

print(f"\nFraud Detection Result:")
print(f"  Fraud Probability: {fraud_probability:.2%}")
print(f"  Decision: {'⚠️ FLAGGED AS FRAUD' if prediction == 1 else '✅ APPROVED'}")

if fraud_probability > 0.7:
    print("  Action: Block transaction, notify customer")
elif fraud_probability > 0.3:
    print("  Action: Request 2FA verification")
else:
    print("  Action: Allow transaction")

# Demonstrate how sample weights change during training
print("\n" + "="*60)
print("HOW SAMPLE WEIGHTS EVOLVE DURING TRAINING")
print("="*60)

# Train a small AdaBoost to show weight progression
ada_debug = AdaBoostClassifier(n_estimators=3, random_state=42)
ada_debug.fit(X_train[:100], y_train[:100])

# Show that misclassified samples get higher weights
print("\nFirst 10 training samples - Weight progression:")
for i in range(3):  # Show first 3 estimators
    weights = ada_debug.estimator_weights_[i]
    print(f"  Estimator {i+1} weight (alpha): {weights:.4f}")