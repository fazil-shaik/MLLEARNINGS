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

#polynomial regression

from sklearn.preprocessing import PolynomialFeatures
np.random.seed(42)


X = np.random.rand(400,3) * 10
y = 2*X[:,0]**2 + 3*X[:,1]**2 + 4*X[:,2]**2 + np.random.randn(400) * 10

#splitting data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection
PolynomialMOdel = PolynomialFeatures(degree=3)
X_train_poly = PolynomialMOdel.fit_transform(X_train)
X_test_poly = PolynomialMOdel.transform(X_test)

#fitting model
PolyModel = LinearRegression()
PolyModel.fit(X_train_poly,y_train) 

#predicting model
y_poly_pred = PolyModel.predict(X_test_poly)

y_test_score = mean_squared_error(y_test,y_poly_pred)
x_test_score = mean_squared_error(y_test,y_poly_pred)

print(f"Polynomial Regression mse Score: {y_test_score}")
print(f"Polynomial Regression mse Score: {y_test_score}")
#model evaluation
mse_poly = mean_squared_error(y_test,y_poly_pred)
r2_poly = r2_score(y_test,y_poly_pred)
print(f"Polynomial Regression Mean Squared Error: {mse_poly}")
print(f"Polynomial Regression R-squared Score: {r2_poly}")

#plotting model
plt.figure()
plt.scatter(y_test, y_poly_pred, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Polynomial Regression: Actual vs Predicted')
plt.legend()
plt.show()

#ridge regression

from sklearn.linear_model import Ridge

np.random.seed(42)


X = np.random.rand(300,3) * 10
y = 2*X[:,0]+ 3*X[:,1]+ 4*X[:,2] + np.random.randn(300) * 10

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection
RidgeModel = Ridge(alpha=5)
RidgeModel.fit(X_train,y_train)

#Model prediction
y_ridge_pred = RidgeModel.predict(X_test)

print(f"Ridge Regression Mean Squared Error: {mean_squared_error(y_test,y_ridge_pred)}" )
print(f"Ridge Regression R-squared Score: {r2_score(y_test,y_ridge_pred)}" )
print(f"Ridge Regression Coefficients: {RidgeModel.coef_}")
print(f"Ridge Regression Intercept: {RidgeModel.intercept_}")


#model plotting
plt.figure()
plt.scatter(y_test,y_ridge_pred,label='predicted vs actual')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',label='ideal fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Ridge Regression: Actual vs Predicted')
plt.legend()
plt.show()  



#lasso regression
from sklearn.linear_model import Lasso

np.random.seed(42)

X = np.random.rand(300,3) * 10
y = 2*X[:,0]+ 3*X[:,1]+ 4*X[:,2] + np.random.randn(300) * 10


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection

LassoModel = Lasso(alpha=0.1)
LassoModel.fit(X_train,y_train)

#model prediction
y_lasso_pred = LassoModel.predict(X_test)

print(f"Lasso Regression Mean Squared Error: {mean_squared_error(y_test,y_lasso_pred)}" )
print(f"Lasso Regression R-squared Score: {r2_score(y_test,y_lasso_pred)}" )
print(f"Lasso Regression Coefficients: {LassoModel.coef_}")
print(f"Lasso Regression Intercept: {LassoModel.intercept_}")

#model plotting
plt.figure()
plt.scatter(y_test,y_lasso_pred,label='predicted vs actual')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',label='ideal fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Lasso Regression: Actual vs Predicted')
plt.legend()
plt.show()

#elastic net regression
from sklearn.linear_model import ElasticNet

np.random.seed(42)

X = np.random.rand(300,3) * 10
y = 2*X[:,0]+ 3*X[:,1]+ 4*  X[:,2] + np.random.randn(300) * 10

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection
ElasticNetModel = ElasticNet(alpha=0.1,l1_ratio=0.5)
ElasticNetModel.fit(X_train,y_train)
#model prediction
y_elastic_pred = ElasticNetModel.predict(X_test)
print(f"Elastic Net Regression Mean Squared Error: {mean_squared_error(y_test,y_elastic_pred)}" )
print(f"Elastic Net Regression R-squared Score: {r2_score(y_test,y_elastic_pred)}" )
print(f"Elastic Net Regression Coefficients: {ElasticNetModel.coef_}")
print(f"Elastic Net Regression Intercept: {ElasticNetModel.intercept_}")

#model plotting
plt.figure()
plt.scatter(y_test, y_elastic_pred, label='predicted vs actual')
sorted_y_test = np.sort(y_test)
plt.plot(sorted_y_test, sorted_y_test, 'r--', label='ideal fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Elastic Net Regression: Actual vs Predicted')
plt.legend()
plt.show()