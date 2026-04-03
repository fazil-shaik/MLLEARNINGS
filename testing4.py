import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 10000

# Generate features
cap_type = np.random.choice(['metal', 'plastic', 'cork', 'composite', 'aluminum', 'hybrid'], n_samples)
production_speed = np.random.normal(500, 100, n_samples)  # bottles per minute
temperature = np.random.uniform(15, 35, n_samples)  # Celsius
cap_quality_score = np.random.beta(7, 3, n_samples) * 100  # Skewed towards high quality
operator_experience = np.random.exponential(3, n_samples)  # years, skewed (many junior)
humidity = np.random.uniform(30, 90, n_samples)  # %

# Create complex non-linear relationships for failure
failure_probability = []

for i in range(n_samples):
    prob = 0
    
    # Base probability
    prob += 0.05
    
    # Cap type effects (non-linear interactions)
    if cap_type[i] == 'plastic':
        prob += 0.15 if temperature[i] > 28 else 0.05
    elif cap_type[i] == 'cork':
        prob += 0.20 if humidity[i] > 70 else 0.02
    elif cap_type[i] == 'composite':
        prob += 0.10 * (1 - cap_quality_score[i]/100)
    elif cap_type[i] == 'metal':
        prob += 0.08 if production_speed[i] > 550 else 0.01
    elif cap_type[i] == 'aluminum':
        prob += 0.12 * (production_speed[i] / 1000)
    else:  # hybrid
        prob += 0.07
    
    # Production speed interaction with operator experience
    if production_speed[i] > 550 and operator_experience[i] < 2:
        prob += 0.25
    elif production_speed[i] > 600 and operator_experience[i] < 5:
        prob += 0.15
    
    # Temperature threshold effects
    if temperature[i] > 30 and humidity[i] > 80:
        prob += 0.30
    elif temperature[i] < 18:
        prob += 0.10  # Cold makes plastic caps brittle
    
    # Quality score interaction
    if cap_quality_score[i] < 40:
        prob += 0.35
    elif cap_quality_score[i] < 60:
        prob += 0.15
    
    # Add random noise
    prob += np.random.normal(0, 0.05)
    
    # Cap probability between 0 and 1
    failure_probability.append(np.clip(prob, 0, 1))

# Generate target (1 = failure, 0 = success)
failure = np.random.binomial(1, failure_probability)

# Create DataFrame
df = pd.DataFrame({
    'cap_type': cap_type,
    'production_speed': production_speed,
    'temperature': temperature,
    'cap_quality_score': cap_quality_score,
    'operator_experience': operator_experience,
    'humidity': humidity,
    'failure': failure
})

# Encode categorical variable
le = LabelEncoder()
df['cap_type_encoded'] = le.fit_transform(df['cap_type'])

# Prepare features
feature_columns = ['cap_type_encoded', 'production_speed', 'temperature', 
                   'cap_quality_score', 'operator_experience', 'humidity']
X = df[feature_columns]
y = df['failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dataset shape: {df.shape}")
print(f"Failure rate: {df['failure'].mean():.2%}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")


ModelClassiifier = RandomForestClassifier(n_estimators=100,random_state=42)
ModelClassiifier.fit(X_train, y_train)

y_pred = ModelClassiifier.predict(X_test)
y_proba = ModelClassiifier.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

plt.style.use('dark_background')

fig,axes = plt.subplots(1,3,figsize=(10,5))

axes[0].bar(X.columns, ModelClassiifier.feature_importances_, color="#3f4f56")
axes[0].set_xlabel("Features", color='white')
axes[0].set_ylabel("Coefficient Value", color='white') 
axes[0].set_title("Feature Coefficients - Logistic Regression", color='white')
axes[0].tick_params(colors='white')
axes[0].grid(True, alpha=0.2)

axes[1].scatter(y_test, y_proba, color="#3f4f56", alpha=0.7)
axes[1].set_xlabel("True Labels", color='white')
axes[1].set_ylabel("Predicted Probabilities", color='white')
axes[1].set_title("True vs Predicted Probabilities", color='white')
axes[1].tick_params(colors='white')
axes[1].grid(True, alpha=0.2)   


#actual vs predicted
axes[2].scatter(y_test, y_pred, color="#3f4f56", alpha=0.7)
axes[2].plot([0, 1], [0, 1], color='red', linestyle='--')  # Add diagonal line for reference
axes[2].set_xlabel("True Labels", color='white')
axes[2].set_ylabel("Predicted Labels", color='white')
axes[2].set_title("True vs Predicted Labels", color='white')
axes[2].tick_params(colors='white')
axes[2].grid(True, alpha=0.2)

#residual plot
residuals = y_test - y_proba
plt.figure(figsize=(6,4))
plt.scatter(y_proba, residuals, color="#3f4f56", alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Probabilities", color='white')
plt.ylabel("Residuals", color='white')
plt.title("Residual Plot", color='white')
plt.tick_params(colors='white')
plt.grid(True, alpha=0.2)

plt.show()




from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Cost assumptions (in dollars)
cost_false_positive = 0.50   # Extra inspection cost
cost_false_negative = 15.00  # Customer complaint + spoiled beer

total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
savings_vs_random = total_cost - (len(y_test) * 0.10 * cost_false_negative)  # Assuming 10% baseline

print("\n" + "="*50)
print("BUSINESS IMPACT ANALYSIS")
print("="*50)
print(f"False Positives: {fp} bottles (unnecessary inspection cost: ${fp * cost_false_positive:.2f})")
print(f"False Negatives: {fn} bottles (missed failures cost: ${fn * cost_false_negative:.2f})")
print(f"Total Quality Cost: ${total_cost:.2f}")
print(f"Estimated Savings vs Random Inspection: ${savings_vs_random:.2f}")

# Precision-Recall tradeoff
precisions = []
recalls = []
thresholds = np.arange(0.1, 0.95, 0.05)

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
plt.xlabel('Probability Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Tradeoff for Production Line')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axvline(x=0.5, color='gray', linestyle='--', label='Default Threshold (0.5)')

# Find optimal threshold (maximize F1)
f1_scores = 2 * (np.array(precisions) * np.array(recalls)) / (np.array(precisions) + np.array(recalls) + 1e-8)
optimal_idx = np.argmax(f1_scores)
plt.axvline(x=thresholds[optimal_idx], color='green', linestyle='--', 
           label=f'Optimal Threshold ({thresholds[optimal_idx]:.2f})')
plt.legend()
plt.show()

print(f"\nOptimal probability threshold: {thresholds[optimal_idx]:.2f}")
print(f"  → At this threshold, F1 score = {f1_scores[optimal_idx]:.3f}")
print(f"  → Precision = {precisions[optimal_idx]:.3f}, Recall = {recalls[optimal_idx]:.3f}")


from sklearn.metrics import confusion_matrix
import numpy as np

y_actual = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # 10 bottles
y_pred =    np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0])  # Model predictions

cm = confusion_matrix(y_actual, y_pred)
print(cm)