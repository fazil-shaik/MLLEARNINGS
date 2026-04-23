import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression  # Classification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('./ecomdata.csv')

# Drop ID and date columns (not features)
X = dataset.drop(['churned', 'customer_id', 'registration_date'], axis=1)
y = dataset['churned']

# Encode categorical features WHILE X is still a DataFrame
categorical_cols = ['membership_tier', 'preferred_category', 'preferred_device', 
                    'preferred_payment_method', 'acquisition_channel', 'country', 'gender']

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))  # Handle NaN safely
    le_dict[col] = le

# Now convert to NumPy for sklearn
X = X.values
y = y.values

print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_linear_pred = linear_model.predict(X_test)

print("Linear Regression R^2 Score:", linear_model.score(X_test, y_test))
print("precision_score:", precision_score(y_test, y_linear_pred.round(), zero_division=0))


#plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_linear_pred, alpha=0.5)
plt.xlabel('Actual Churn')
plt.ylabel('Predicted Churn (Linear Regression)')
plt.title('Actual vs Predicted Churn (Linear Regression)')  
plt.subplot(1, 2, 2)
sns.histplot(y_linear_pred, bins=20, kde=True)
plt.xlabel('Predicted Churn (Linear Regression)')
plt.title('Distribution of Predicted Churn (Linear Regression)')
plt.tight_layout()
plt.show()


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Churn probability

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plotting
plt.figure(figsize=(12, 5))

# Plot 1: Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Plot 2: Predicted Probability Distribution
plt.subplot(1, 2, 2)
plt.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.6, label='Not Churned', color='green')
plt.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.6, label='Churned', color='red')
plt.xlabel('Churn Probability')
plt.ylabel('Count')
plt.title('Predicted Probability Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# Feature importance (coefficients)
# feature_names = ['total_spent', 'num_purchases', 'avg_purchase_value', 'membership_tier', 'preferred_category', 'preferred_device', 
#                  'preferred_payment_method', 'acquisition_channel', 'country', 'gender']
# coefficients = model.coef_[0]
# feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
# feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()
# feature_importance = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_importance, palette='viridis')
# plt.title('Feature Importance (Logistic Regression Coefficients)')
# plt.xlabel('Absolute Coefficient Value')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()      

#lasso regression
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1, random_state=42)
lasso_model.fit(X_train, y_train)

y_lasso_pred = lasso_model.predict(X_test)

print("Lasso Regression R^2 Score:", lasso_model.score(X_test, y_test))

# Plotting Lasso results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_lasso_pred, alpha=0.5)
plt.xlabel('Actual Churn')
plt.ylabel('Predicted Churn (Lasso Regression)')
plt.title('Actual vs Predicted Churn (Lasso Regression)')
plt.subplot(1, 2, 2)
sns.histplot(y_lasso_pred, bins=20, kde=True)
plt.xlabel('Predicted Churn (Lasso Regression)')
plt.title('Distribution of Predicted Churn (Lasso Regression)')
plt.tight_layout()
plt.show()