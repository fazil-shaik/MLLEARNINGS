import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, StackingClassifier, 
                              VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Create synthetic fraud detection dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    n_clusters_per_class=1,
    weights=[0.995, 0.005],
    flip_y=0.01,
    random_state=42
)

feature_names = [f'feature_{i}' for i in range(20)]
fraud_columns = ['amount', 'hour_of_day', 'location_risk', 'transaction_velocity',
                 'device_risk', 'prev_fraud_flags', 'same_merchant', 'ip_risk',
                 'card_age_days', 'transaction_type', 'currency_change', 
                 'time_since_last', 'purchase_category', 'shipping_match',
                 'email_age_days', 'purchase_frequency', 'avg_transaction_value',
                 'metadata_hash', 'session_duration', 'click_pattern']
X = pd.DataFrame(X, columns=fraud_columns)

print("Dataset Shape:", X.shape)
print(f"Fraud Rate: {y.mean():.3%}")
print(f"\nFirst 5 transactions:\n{X.head()}")
print(f"\nTarget distribution:\n{pd.Series(y).value_counts()}")

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples ({y_train.mean():.3%} fraud)")
print(f"Test set: {X_test.shape[0]} samples ({y_test.mean():.3%} fraud)")

# Step 3: Define individual models
decision_tree = DecisionTreeClassifier(
    max_depth=5, 
    min_samples_split=20,
    class_weight='balanced',
    random_state=42
)

logistic_regression = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# Step 4: Define Ensemble Methods

# 4.1 BAGGING: Random Forest
print("\n" + "="*60)
print("4.1 RANDOM FOREST (Bagging)")
print("="*60)

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

# 4.2 BOOSTING: Gradient Boosting
print("\n" + "="*60)
print("4.2 GRADIENT BOOSTING")
print("="*60)

gradient_boost = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    subsample=0.8,
    random_state=42
)

# 4.3 BOOSTING: AdaBoost
print("\n" + "="*60)
print("4.3 ADABOOST")
print("="*60)

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=50,
    learning_rate=1.0,
    # algorithm='SAMME',
    random_state=42
)

# 4.4 VOTING CLASSIFIER (Only Soft Voting - has predict_proba)
print("\n" + "="*60)
print("4.4 VOTING CLASSIFIER (Soft Voting)")
print("="*60)

voting_soft = VotingClassifier(
    estimators=[
        ('dt', decision_tree),
        ('lr', logistic_regression),
        ('rf', random_forest),
        ('gb', gradient_boost)
    ],
    voting='soft',  # Soft voting uses probabilities
    weights=[1, 1, 2, 2]  # Give more weight to RF and GB
)

# 4.5 STACKING (Meta-ensemble)
print("\n" + "="*60)
print("4.5 STACKING CLASSIFIER")
print("="*60)

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

meta_learner = LogisticRegression()

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba'
)

# Step 5: Train and evaluate all models WITH PROPER ERROR HANDLING
models = {
    'Decision Tree (Base)': decision_tree,
    'Random Forest (Bagging)': random_forest,
    'Gradient Boosting': gradient_boost,
    'AdaBoost': adaboost,
    'Voting (Soft)': voting_soft,
    'Stacking': stacking_clf
}

results = []

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Handle predict_proba carefully - some models might not have it
    try:
        # Check if model has predict_proba
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            # Fall back to decision function or just use predictions
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_score)
            else:
                roc_auc = None
                print(f"  Warning: {name} doesn't support probability predictions")
    except Exception as e:
        print(f"  Warning: Could not compute ROC-AUC for {name}: {e}")
        roc_auc = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Cross-validation score (using ROC-AUC when possible, else accuracy)
    try:
        if roc_auc is not None:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            cv_metric = f"{cv_scores.mean():.3f}±{cv_scores.std():.3f}"
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_metric = f"{cv_scores.mean():.3f}±{cv_scores.std():.3f}"
    except:
        cv_metric = "N/A"
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Fraud Catch Rate)': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc if roc_auc else "N/A",
        'CV Score': cv_metric
    })
    
    print(f"\nResults for {name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (Frauds caught): {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    True Negatives: {cm[0,0]:,} | False Positives: {cm[0,1]:,}")
    print(f"    False Negatives: {cm[1,0]:,} | True Positives: {cm[1,1]:,}")
    
    # For fraud detection, calculate additional metrics
    if cm[1,1] + cm[1,0] > 0:  # Avoid division by zero
        fraud_catch_rate = cm[1,1] / (cm[1,1] + cm[1,0])
        print(f"  Fraud Catch Rate: {fraud_catch_rate:.2%}")
        
        # False Positive Rate (alerts per legit transaction)
        false_alert_rate = cm[0,1] / (cm[0,0] + cm[0,1])
        print(f"  False Alert Rate: {false_alert_rate:.2%}")

# Step 6: Compare all models
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL COMPARISON - ALL MODELS")
print("="*60)
print(results_df.to_string(index=False))

# Step 7: Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Performance comparison (excluding N/A values)
metrics_to_plot = ['Precision', 'Recall (Fraud Catch Rate)', 'F1-Score']
plot_data = results_df.set_index('Model')[metrics_to_plot].copy()
# Convert any non-numeric to NaN
plot_data = plot_data.apply(pd.to_numeric, errors='coerce')
plot_data.plot(kind='bar', ax=axes[0, 0], colormap='viridis', edgecolor='black')
axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Feature importance from Random Forest
rf_feature_importance = pd.DataFrame({
    'feature': fraud_columns,
    'importance': random_forest.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[0, 1].barh(rf_feature_importance['feature'], rf_feature_importance['importance'])
axes[0, 1].set_title('Top 10 Features for Fraud Detection (Random Forest)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].invert_yaxis()

# Plot 3: Confusion Matrix for Stacking (best ensemble)
stacking_pred = stacking_clf.predict(X_test)
stacking_cm = confusion_matrix(y_test, stacking_pred)
sns.heatmap(stacking_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Stacking Confusion Matrix', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xticklabels(['Legit', 'Fraud'])
axes[1, 0].set_yticklabels(['Legit', 'Fraud'])

# Add percentages to confusion matrix
total = stacking_cm.sum()
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j+0.5, i+0.7, f'{stacking_cm[i,j]/total:.1%}', 
                       ha='center', va='center', color='red', fontweight='bold')

# Plot 4: Cross-validation scores comparison
cv_scores_numeric = []
for score in results_df['CV Score']:
    if score != "N/A":
        try:
            cv_scores_numeric.append(float(score.split('±')[0]))
        except:
            cv_scores_numeric.append(0)
    else:
        cv_scores_numeric.append(0)

axes[1, 1].bar(range(len(models)), cv_scores_numeric, alpha=0.7, color='steelblue')
axes[1, 1].set_xticks(range(len(models)))
axes[1, 1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[1, 1].set_title('Cross-Validation Score (5-fold)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_ylim([0.5, 1])
axes[1, 1].grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(cv_scores_numeric):
    axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Step 8: Detailed analysis of the best model (usually Stacking or Gradient Boosting)
print("\n" + "="*60)
print("DEEP DIVE: BEST PERFORMING ENSEMBLE")
print("="*60)

# Find the best model by F1-score
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"F1-Score: {results_df.loc[results_df['Model'] == best_model_name, 'F1-Score'].values[0]:.4f}")

if best_model_name == 'Stacking':
    best_predictions = stacking_pred
else:
    best_predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, best_predictions, target_names=['Legit', 'Fraud']))

# Calculate business impact
fraud_rate = y_test.mean()
total_transactions = len(y_test)
actual_frauds = y_test.sum()
frauds_detected = (best_predictions[y_test == 1]).sum()
fraud_catch_rate = frauds_detected / actual_frauds if actual_frauds > 0 else 0
false_positives = confusion_matrix(y_test, best_predictions)[0, 1]

print(f"\nBusiness Impact Simulation for {best_model_name}:")
print(f"  Total transactions in test set: {total_transactions:,}")
print(f"  Actual fraud cases: {actual_frauds:,} ({fraud_rate:.2%})")
print(f"  Frauds detected: {frauds_detected:,} ({fraud_catch_rate:.2%})")
print(f"  Frauds missed: {actual_frauds - frauds_detected:,}")
print(f"  False alarms (legit flagged as fraud): {false_positives:,}")

# Estimated cost savings (assuming $500 per fraud, $5 per false alarm)
saved_per_fraud = 500
cost_per_false_alarm = 5
total_savings = (frauds_detected * saved_per_fraud) - (false_positives * cost_per_false_alarm)
print(f"  Estimated net savings: ${total_savings:,.0f}")

# Step 9: Feature importance for tree-based models
print("\n" + "="*60)
print("ENSEMBLE INSIGHTS - WHY THEY WORK")
print("="*60)

# Compare individual vs ensemble performance
base_models = ['Decision Tree (Base)', 'Random Forest (Bagging)', 'Gradient Boosting']
ensemble_models = ['Voting (Soft)', 'Stacking']

base_avg_f1 = results_df[results_df['Model'].isin(base_models)]['F1-Score'].mean()
ensemble_avg_f1 = results_df[results_df['Model'].isin(ensemble_models)]['F1-Score'].mean()

print(f"\nAverage F1-Score for Base Models: {base_avg_f1:.4f}")
print(f"Average F1-Score for Ensemble Models: {ensemble_avg_f1:.4f}")
print(f"Improvement from Ensembles: {(ensemble_avg_f1 - base_avg_f1):.4f} ({(ensemble_avg_f1/base_avg_f1 - 1)*100:.1f}%)")

print("\n" + "="*60)
print("KEY TAKEAWAYS")
print("="*60)
print("""
1. Bagging (Random Forest):
   - Reduces variance by averaging many trees
   - Handles imbalanced data well (class_weight='balanced')
   - Provides feature importance for interpretability
   - Good baseline ensemble

2. Boosting (Gradient Boosting, AdaBoost):
   - Sequentially focuses on hard-to-classify fraud cases
   - Achieves high recall (catches more frauds)
   - Sensitive to noisy data but powerful for fraud detection
   - Can overfit if too many estimators

3. Voting (Soft):
   - Simple combination of diverse models
   - Weights allow emphasizing better performers
   - Requires all models to support predict_proba
   - Robust and stable predictions

4. Stacking:
   - Learns optimal combination through meta-learner
   - Most powerful but computationally expensive
   - Cross-validation prevents overfitting
   - Best for peak performance when data allows

PRACTICAL RECOMMENDATIONS:
- Start with Random Forest (robust, interpretable)
- For imbalanced problems, use Gradient Boosting
- Use Soft Voting when models have different strengths
- Use Stacking only for final production models (computational cost)
- Always validate with cross-validation
""")