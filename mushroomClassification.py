from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


Dataframe = pd.read_csv('./mushrooms.csv')

print(Dataframe.head(),Dataframe.shape)


trandformed_data = pd.get_dummies(Dataframe)
X = trandformed_data.drop('class_e', axis=1)
y = trandformed_data['class_e']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#logistic regression model
LogisticModel = LogisticRegression(max_iter=50000,solver='lbfgs')
LogisticModel.fit(X_train,y_train)


#evaluation
y_pred = LogisticModel.predict(X_test)

y_train_pred = LogisticModel.predict(X_train)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_pred))
print("r2_score:", LogisticModel.score(X_test, y_test))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#plotting results
plt.figure(figsize=(10, 6))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=100, label='Actual')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=100, marker='X', label='Predicted')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('Actual vs Predicted Labels')
plt.legend()
plt.show()

new_values =np.random.rand(1, X.shape[1])  # Generate random values for prediction
new_prediction = LogisticModel.predict(new_values)
print("New Prediction:", new_prediction)


plt.figure(figsize=(10, 6))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=100, label='Actual')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=100, marker='X', label='Predicted')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('Actual vs Predicted Labels')
plt.legend()
plt.show()  


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # Load dataset
# data = load_breast_cancer()
# X = data.data
# y = data.target

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model
# model = LogisticRegression(max_iter=50000, solver='lbfgs')
# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Predicted labels:", y_pred)

# #plotting the resilts 
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=100, label='Actual')
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=100, marker='X', label='Predicted')
# plt.xlabel(data.feature_names[0])
# plt.ylabel(data.feature_names[1])
# plt.title('Actual vs Predicted Labels')
# plt.legend()
# plt.show()


