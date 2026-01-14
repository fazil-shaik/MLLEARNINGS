import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import r2_score,mean_squared_error

#data preprocessing 
res = np.random.seed(7)


n_train = 12
X_train = np.sort(np.random.uniform(-3,3,size=n_train)).reshape(-1,1)


y_true_train = 0.5*X_train.squeeze()**3*X_train.squeeze()

#adding random guassian values with mean 0 and standard deviration-3.0
y_train = y_true_train+np.random.normal(0,3.0,size=n_train)

#test the model
X_test = np.linspace(-3,3,200).reshape(-1,1)
y_true_test= 0.5 * X_test.squeeze()**3 - 2*X_test.squeeze() 


y_test = y_true_test


#our goal is to increase overfitting 
degree=15

#linear modle bundling model pipeline

linear_model = Pipeline([
    ("poly",PolynomialFeatures(degree=degree,include_bias=False)),
    ("linear",LinearRegression())
])

ridege_model = Pipeline([
    ("poly",PolynomialFeatures(degree=degree,include_bias=False)),
    ("ridge",Ridge(alpha=10.0))
])

#train the model
linear_model.fit(X_train,y_train)
ridege_model.fit(X_train,y_train)

#predict the model
y_pred_linear_test = linear_model.predict(X_test)
y_pred_ridge_test = ridege_model.predict(X_test)

#mse
mse_error_linear = mean_squared_error(y_train,linear_model.predict(X_train))

mse_error_ridge = mean_squared_error(y_train,ridege_model.predict(X_train))

mean_square_linear_test = mean_squared_error(y_test,y_pred_linear_test)
mean_square_ridge_test = mean_squared_error(y_test,y_pred_ridge_test)

plt.figure(figsize=(10,5))
plt.scatter(X_train,y_train,label='Actual data')
plt.plot(X_test,y_true_test,label='true curves')
plt.plot(X_test,y_pred_linear_test,label=f"linear regression with {degree}")
plt.plot(X_test,y_pred_ridge_test,label=f"Ridge regression with {degree}")
plt.title("Overfitting demo")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
