# from sklearn.model_selection import train_test_split
# import pandas as pd
# from sklearn.metrics import r2_score,accuracy_score,classification_report
# from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score,cross_validate,KFold
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
# from sklearn.preprocessing import label_binarize
# import seaborn as sns

# churn = pd.read_csv('./datalab.csv')
# churn.head()

# print(churn.shape)
# print(churn.columns)
# print(churn.info())


# X = churn.drop('Churn',axis=1)
# y = churn.Churn

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size
# =0.2, random_state=42)



# pipline = Pipeline([
#     ('scaler',StandardScaler()),
#     ('classifier',DecisionTreeClassifier(random_state=42))
# ])

# pipline.fit(X_train,y_train)


# y_predict = pipline.predict(X_test)
# print(classification_report(y_predict, y_test))

# y_pred_proba_pipeline = pipline.predict_proba(X_test)[:, 1]

# cv_scores = cross_val_score(pipline,X=X,y=y,cv=5)

# print(f"Cross-validation scores: {cv_scores}")
# print(f"Mean CV accuracy: {np.mean(cv_scores):.2f}")



# from sklearn.ensemble import BaggingClassifier


# bagging_class = BaggingClassifier(pipline,n_estimators=50,random_state=42)
# bagging_class.fit(X_train, y_train)


# y_pred = bagging_class.predict(X_test)

# # Classification Report
# print("\n","======"*10)
# print("\nBagging classification report")
# print(classification_report(y_pred, y_test))


# #cv scoresBagging
# print("\n","======"*10)
# cv_bagging = cross_val_score(bagging_class,X,y,cv=5)
# print(f"Bagging Cross-validation scores: {cv_bagging}")
# print(f"Mean Bagging CV accuracy: {np.mean(cv_bagging):.2f}")



# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
# # Confusion Matrix for Pipeline
# cm_pipeline = confusion_matrix(y_test, y_predict)
# sns.heatmap(cm_pipeline, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
#             cbar=False, xticklabels=['No Churn', 'Churn'], 
#             yticklabels=['No Churn', 'Churn'])
# axes[0].set_title('Pipeline - Confusion Matrix')
# axes[0].set_ylabel('Actual')
# axes[0].set_xlabel('Predicted')
 
# # Confusion Matrix for Bagging
# cm_bagging = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm_bagging, annot=True, fmt='d', cmap='Greens', ax=axes[1],
#             cbar=False, xticklabels=['No Churn', 'Churn'], 
#             yticklabels=['No Churn', 'Churn'])
# axes[1].set_title('Bagging - Confusion Matrix')
# axes[1].set_ylabel('Actual')
# axes[1].set_xlabel('Predicted')
 
# plt.tight_layout()
# plt.show()
 
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
# # ROC for Pipeline
# fpr_pipe, tpr_pipe, _ = roc_curve(y_test, y_pred_proba_pipeline)
# roc_auc_pipe = auc(fpr_pipe, tpr_pipe)
 
# axes[0].plot(fpr_pipe, tpr_pipe, color='blue', lw=2, 
#              label=f'ROC curve (AUC = {roc_auc_pipe:.3f})')
# axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
# axes[0].set_xlabel('False Positive Rate')
# axes[0].set_ylabel('True Positive Rate')
# axes[0].set_title('Pipeline - ROC Curve')
# axes[0].legend(loc="lower right")
# axes[0].grid(alpha=0.3)
 
# # ROC for Bagging
# y_pred_proba_bagging = bagging_class.predict_proba(X_test)[:, 1]
# fpr_bag, tpr_bag, _ = roc_curve(y_test, y_pred_proba_bagging)
# roc_auc_bag = auc(fpr_bag, tpr_bag)
 
# axes[1].plot(fpr_bag, tpr_bag, color='green', lw=2, 
#              label=f'ROC curve (AUC = {roc_auc_bag:.3f})')
# axes[1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
# axes[1].set_xlabel('False Positive Rate')
# axes[1].set_ylabel('True Positive Rate')
# axes[1].set_title('Bagging - ROC Curve')
# axes[1].legend(loc="lower right")
# axes[1].grid(alpha=0.3)
 
# plt.tight_layout()
# plt.show()
 
# # METHOD 3: ACTUAL vs PREDICTED (Bar Chart - Better for small datasets)
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
# # Pipeline
# x_pos = np.arange(len(y_test))
# width = 0.35
# axes[0].bar(x_pos - width/2, y_test.values, width, label='Actual', alpha=0.7)
# axes[0].bar(x_pos + width/2, y_predict, width, label='Predicted', alpha=0.7)
# axes[0].set_xlabel('Sample Index')
# axes[0].set_ylabel('Churn (0=No, 1=Yes)')
# axes[0].set_title('Pipeline - Actual vs Predicted')
# axes[0].legend()
# axes[0].set_ylim(-0.1, 1.1)
 
# # Bagging
# axes[1].bar(x_pos - width/2, y_test.values, width, label='Actual', alpha=0.7)
# axes[1].bar(x_pos + width/2, y_pred, width, label='Predicted', alpha=0.7)
# axes[1].set_xlabel('Sample Index')
# axes[1].set_ylabel('Churn (0=No, 1=Yes)')
# axes[1].set_title('Bagging - Actual vs Predicted')
# axes[1].legend()
# axes[1].set_ylim(-0.1, 1.1)
 
# plt.tight_layout()
# plt.show()
 
# # METHOD 4: MISCLASSIFIED SAMPLES (Highlight errors)
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
# # Pipeline - Show which samples were wrong
# pipeline_errors = y_test.values != y_predict
# axes[0].scatter(range(len(y_test)), y_test.values, label='Actual', s=100, alpha=0.6)
# axes[0].scatter(np.where(pipeline_errors)[0], y_predict[pipeline_errors], 
#                 label='Misclassified', s=150, marker='X', color='red', edgecolors='darkred', linewidth=2)
# axes[0].set_xlabel('Sample Index')
# axes[0].set_ylabel('Churn (0=No, 1=Yes)')
# axes[0].set_title(f'Pipeline - Misclassified: {pipeline_errors.sum()} out of {len(y_test)}')
# axes[0].legend()
# axes[0].set_ylim(-0.1, 1.1)
# axes[0].grid(alpha=0.3)
 
# # Bagging - Show which samples were wrong
# bagging_errors = y_test.values != y_pred
# axes[1].scatter(range(len(y_test)), y_test.values, label='Actual', s=100, alpha=0.6)
# axes[1].scatter(np.where(bagging_errors)[0], y_pred[bagging_errors], 
#                 label='Misclassified', s=150, marker='X', color='red', edgecolors='darkred', linewidth=2)
# axes[1].set_xlabel('Sample Index')
# axes[1].set_ylabel('Churn (0=No, 1=Yes)')
# axes[1].set_title(f'Bagging - Misclassified: {bagging_errors.sum()} out of {len(y_test)}')
# axes[1].legend()
# axes[1].set_ylim(-0.1, 1.1)
# axes[1].grid(alpha=0.3)
 
# plt.tight_layout()
# plt.show()
 
# # METHOD 5: COMPARISON METRICS (Simple visualization of performance)
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
 
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
# pipeline_scores = [
#     accuracy_score(y_test, y_predict),
#     precision_score(y_test, y_predict),
#     recall_score(y_test, y_predict),
#     f1_score(y_test, y_predict)
# ]
# bagging_scores = [
#     accuracy_score(y_test, y_pred),
#     precision_score(y_test, y_pred),
#     recall_score(y_test, y_pred),
#     f1_score(y_test, y_pred)
# ]
 
# x = np.arange(len(metrics))
# width = 0.35
 
# fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar(x - width/2, pipeline_scores, width, label='Pipeline', alpha=0.8)
# bars2 = ax.bar(x + width/2, bagging_scores, width, label='Bagging', alpha=0.8)
 
# ax.set_ylabel('Score')
# ax.set_title('Model Comparison: Pipeline vs Bagging')
# ax.set_xticks(x)
# ax.set_xticklabels(metrics)
# ax.legend()
# ax.set_ylim([0, 1.1])
# ax.grid(axis='y', alpha=0.3)
 
# # Add value labels on bars
# for bars in [bars1, bars2]:
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)
 
# plt.tight_layout()
# plt.show()






#Boosting 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

n_samples = 50000

credit_score = np.random.normal(680, 70, n_samples).clip(300, 850)
annual_income = np.random.lognormal(10.5, 0.8, n_samples) / 1000
debt_to_income = np.random.beta(2, 5, n_samples) * 0.8
employment_years = np.random.exponential(5, n_samples).clip(0, 40)
loan_amount = np.random.lognormal(10, 0.7, n_samples) / 1000
past_defaults = np.random.poisson(0.2, n_samples).clip(0, 3)
num_credit_lines = np.random.poisson(8, n_samples).clip(1, 25)

risk_score = (
    (credit_score < 620) * (debt_to_income > 0.4) * 3 +
    (past_defaults > 0) * (loan_amount > 50) * 2 +
    (debt_to_income > 0.5) * 1.5 +
    (employment_years < 1) * 1.0 +
    np.random.normal(0, 0.5, n_samples)
)

default_prob = 1 / (1 + np.exp(-risk_score))
default = (default_prob > 0.5).astype(int)

df = pd.DataFrame({
    'credit_score': credit_score,
    'annual_income': annual_income,
    'debt_to_income': debt_to_income,
    'employment_years': employment_years,
    'loan_amount': loan_amount,
    'past_defaults': past_defaults,
    'num_credit_lines': num_credit_lines,
    'default': default
})

income_missing_mask = np.random.random(len(df)) < 0.15
df.loc[income_missing_mask, 'annual_income'] = np.nan

feature_cols = ['credit_score', 'annual_income', 'debt_to_income', 
                'employment_years', 'loan_amount', 'past_defaults', 
                'num_credit_lines']
X = df[feature_cols]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

gb_model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train, y_train)




imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_imp, y_train)

gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
gb_pred = (gb_pred_proba > 0.5).astype(int)
lr_pred_proba = lr_model.predict_proba(X_test_imp)[:, 1]
lr_pred = (lr_pred_proba > 0.5).astype(int)

tn_gb, fp_gb, fn_gb, tp_gb = confusion_matrix(y_test, gb_pred).ravel()
tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_test, lr_pred).ravel()

gb_auc = roc_auc_score(y_test, gb_pred_proba)
lr_auc = roc_auc_score(y_test, lr_pred_proba)

gb_recall = tp_gb / (tp_gb + fn_gb) if (tp_gb + fn_gb) > 0 else 0
lr_recall = tp_lr / (tp_lr + fn_lr) if (tp_lr + fn_lr) > 0 else 0

loss_bad_loans_gb = fn_gb * 10000
loss_false_rejections_gb = fp_gb * 500
profit_good_gb = tn_gb * 500
net_profit_gb = profit_good_gb - (loss_bad_loans_gb + loss_false_rejections_gb)

loss_bad_loans_lr = fn_lr * 10000
loss_false_rejections_lr = fp_lr * 500
profit_good_lr = tn_lr * 500
net_profit_lr = profit_good_lr - (loss_bad_loans_lr + loss_false_rejections_lr)

print("LOAN DEFAULT PREDICTION RESULTS")
print("="*50)

print("\nHISTOGRAM-BASED GRADIENT BOOSTING RESULTS")
print("-"*30)
print(f"ROC-AUC: {gb_auc:.4f}")
print(f"Recall (default detection): {gb_recall:.4f}")
print(f"Missed defaults: {fn_gb}")
print(f"False rejections: {fp_gb}")
print(f"Net profit: ${net_profit_gb:,.0f}")

print("\nLOGISTIC REGRESSION RESULTS")
print("-"*30)
print(f"ROC-AUC: {lr_auc:.4f}")
print(f"Recall (default detection): {lr_recall:.4f}")
print(f"Missed defaults: {fn_lr}")
print(f"False rejections: {fp_lr}")
print(f"Net profit: ${net_profit_lr:,.0f}")

print("\nIMPROVEMENT WITH BOOSTING")
print("-"*30)
print(f"ROC-AUC improvement: {(gb_auc - lr_auc)*100:.1f}%")
print(f"Defaults detected: {fn_lr - fn_gb} more")
print(f"Profit increase: ${net_profit_gb - net_profit_lr:,.0f}")

print("\nFEATURE IMPORTANCE")
print("-"*30)

perm_importance = permutation_importance(gb_model,X,y,n_repeats=10,random_state=42)

importacnes = perm_importance.importances_mean

print("importacnes are ",importacnes)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importacnes
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.3f}")



plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importances - Gradient Boosting')
plt.tight_layout()
plt.show()



gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
gb_pred = (gb_pred_proba > 0.5).astype(int)

comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': gb_pred,
    'Probability': gb_pred_proba
})

print(comparison_df.head(20))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(range(len(y_test[:200])), y_test[:200], alpha=0.6, label='Actual', s=20)
plt.scatter(range(len(y_test[:200])), gb_pred[:200], alpha=0.6, label='Predicted', s=20, marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Class (0=Good, 1=Default)')
plt.title('Actual vs Predicted Values (First 200 samples)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
errors = (gb_pred != y_test).astype(int)
plt.scatter(range(len(y_test[:200])), y_test[:200], c=errors[:200], cmap='RdYlGn', 
            alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Actual Class')
plt.title('Correct (Green) vs Incorrect (Red) Predictions')
plt.colorbar(label='Prediction Error (0=Correct, 1=Incorrect)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()