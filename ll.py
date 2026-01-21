import numpy as np
from sklearn.linear_model import Lasso,LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

np.random.seed(45)


n_train = 20
n_test = 100





X_train = np.sort(np.random.uniform(-3,3,size=n_train)).reshape(-1,1)
y_true_train = 0.5*X_train.squeeze()**3 - 2 * X_train.squeeze()**2 + X_train.squeeze() + 3
y_train = y_true_train + np.random.normal(0,0.3,size=n_train)


X_test = np.linspace(-3,3,200).reshape(-1,1)
y_true_test = 0.5*X_test.squeeze()**3 -2 * X_test.squeeze()**2 + X_test.squeeze() + 3
y_test = y_true_test

degree = 15

linear_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree,include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression()),
])

lasso_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree,include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Lasso(alpha=0.1,max_iter=51000))
])

linear_model.fit(X_train,y_train)
lasso_model.fit(X_train,y_train)


y_linear_test = linear_model.predict(X_test)
y_lasso_test = lasso_model.predict(X_test)

mse_linear = mean_squared_error(y_test,y_linear_test)
mse_lasso = mean_squared_error(y_test,y_lasso_test)

print(f"Linear Regression Test MSE: {mse_linear:.4f}")
print(f"Lasso Regression Test MSE: {mse_lasso:.4f}")

print("linear model values",y_linear_test)
print("Lasso model values",y_lasso_test)

plt.figure(figsize=(12,6))
plt.scatter(X_train,y_train,color='blue',label='Training Data')
plt.plot(X_test,y_true_test,color='green',label='True Function',linewidth=2)
plt.plot(X_test,y_linear_test,color='red',label='Linear Regression Prediction',linewidth=2)
plt.plot(X_test,y_lasso_test,color='orange',label='Lasso Regression Prediction',linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression vs Lasso Regression')
plt.legend()
plt.show()


print("Training data X:",X_train)







# X = np.sort(np.random.rand(n_train + n_test))[:, np.newaxis]
# y = np.sin(2 * np.pi * X).ravel()
# y[::5] += 3 * (0.5 - np.random.rand(n_train + n_test))
# X_train, X_test = X[:n_train], X[n_train:]
# y_train, y_test = y[:n_train], y[n_train:]

# model = Pipeline([
#     ('poly', PolynomialFeatures(degree=10)),
#     ('scaler', StandardScaler()),
#     ('regressor', Lasso(alpha=0.01))
# ])
# model.fit(X_train, y_train)
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test) 
# mse_train = mean_squared_error(y_train, y_train_pred)
# mse_test = mean_squared_error(y_test, y_test_pred)
# r2_train = r2_score(y_train, y_train_pred)
# r2_test = r2_score(y_test, y_test_pred)
# print(f"Train MSE: {mse_train:.4f}, R2: {r2_train:.4f}")
# print(f"Test MSE: {mse_test:.4f}, R2: {r2_test:.4f}")
# import matplotlib.pyplot as plt
# plt.scatter(X_train, y_train, color='blue', label='Training data')
# plt.scatter(X_test, y_test, color='green', label='Testing data')
# X_all = np.linspace(0, 1, 100)[:, np.newaxis]   
# y_all_pred = model.predict(X_all)
# plt.plot(X_all, y_all_pred, color='red', label='Model prediction')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Lasso Regression with Polynomial Features')
# plt.legend()
# plt.show()
