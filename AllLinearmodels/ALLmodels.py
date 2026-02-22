import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split


np.random.seed(42)

n = 200   # samples
f = 3     # features

# Random features
X = np.random.rand(n, f)

# Define true weights manually
w = np.array([4, -2, 3])
b = 5

# Generate target
y = X @ w + b + np.random.randn(n) * 0.2
#model training and splitting 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection 
LinearModel = LinearRegression()
LinearModel.fit(X_train,y_train)

#model prediction and evaluation
y_Linear_pred = LinearModel.predict(X_test)

#model evaliatin 
mse = mean_squared_error(y_test,y_Linear_pred)
r2 = r2_score(y_test,y_Linear_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

#plotting predctions
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(y_test, y_Linear_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()


#multiple linear regression

np.random.seed(42)

n = 300

# Features
size = np.random.rand(n) * 2000      # house size
bedrooms = np.random.randint(1, 6, n)
age = np.random.rand(n) * 30

X = np.column_stack([size, bedrooms, age])

# True relationship
y = (
    100 * size +
    20000 * bedrooms -
    500 * age +
    np.random.randn(n) * 10000
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#model selection
MultipleModel = LinearRegression()
MultipleModel.fit(X_train,y_train)

#predicting model
y_pred = MultipleModel.predict(X_test)

#plotting and model and evaluating
plt.figure()
plt.scatter(y_test, y_pred, label='Predicted vs Actual')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',label='Ideal Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()
