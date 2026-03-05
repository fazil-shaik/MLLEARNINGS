import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# PART 1: SVM Classification


print("\n===============================")
print("SVM CLASSIFICATION EXAMPLE")
print("===============================\n")

# Load real-world dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM Model
svm_model = SVC(kernel='rbf', C=1)

svm_model.fit(X_train, y_train)

# Prediction
y_pred = svm_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("SVM Classification Accuracy:", accuracy)


# PART 2: SVR Regression

print("\n===============================")
print("SVR REGRESSION EXAMPLE")
print("===============================\n")

# Load dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVR Model
svr_model = SVR(
    kernel='rbf',
    C=100,
    epsilon=0.1
)

svr_model.fit(X_train, y_train)

# Prediction
predictions = svr_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)

print("SVR Mean Squared Error:", mse)


# Visualization

plt.figure()

plt.scatter(y_test, predictions)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")

plt.title("SVR Prediction vs Actual")

plt.show()