import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import plot_tree
from sklearn.linear_model import Lasso,Ridge as ridge_regression

np.random.seed(42)

# X = np.sort(5*np.random.rand(180, 1), axis=0)
# y = np.sin(X).ravel() + np.random.normal(0, 0.5, X.shape[0])

X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.001, X.shape[0])
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   

#Orginal data plotting

plt.scatter(X, y, color='red', label='Data')
plt.title("Synthetic Dataset")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()


regressor = DecisionTreeRegressor(max_depth=3, random_state=42)
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize the results
X_grid = np.arange(0, 5, 0.01).reshape(-1, 1)
y_grid_pred = regressor.predict(X_grid) 

plt.scatter(X, y, color='red', label='Data')
plt.plot(X_grid, y_grid_pred, color='blue', label='DTR Prediction')
plt.title("Decision Tree Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()



plt.figure(figsize=(12, 8))

plot_tree(regressor,
          
feature_names=['Feature'],
          filled=True,
          rounded=True,
          fontsize=10
          )

plt.title("Decision Tree Structure")
plt.show()

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

r2_score_train = r2_score(y_train,y_train_pred)
r2_score_test = r2_score(y_test,y_test_pred)



print(f"Train r2: {r2_score_train:.3f}")
print(f"Test r2: {r2_score_test:.3f}")



train_mse = np.sqrt(mean_squared_error(y_train,y_train_pred))
test_mse = np.sqrt(mean_squared_error(y_test,y_test_pred))


print(f"Train RMSE: {train_mse:.3f}")
print(f"Test RMSE: {test_mse:.3f}")



residuals = y_test-y_test_pred

plt.scatter(y_test_pred,residuals)

plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()




LassoRegression = Lasso(
    alpha=0.2,
    max_iter=1000,
    random_state=42
)

LassoRegression.fit(X_train,y_train)


y_lasso_predict = LassoRegression.predict(X_test)


y_lasso_train_pred = regressor.predict(X_train)
y_lasso_test_pred = regressor.predict(X_test)

r2_score_train_lasso = r2_score(y_train,y_lasso_train_pred)
r2_score_test_lasso = r2_score(y_test,y_lasso_test_pred)

print(f"MSE of train_data {mean_squared_error(y_train,y_lasso_train_pred)}")
print(f"MSE of test data {mean_squared_error(y_test,y_lasso_test_pred)}")

print(f"r2_score of train_data {r2_score_train_lasso}")
print(f"r2_score of test data {r2_score_test_lasso}")


residuals_lasso = y_test-y_lasso_test_pred
plt.scatter(y_lasso_test_pred,residuals_lasso)
plt.axhline(y=0,color='black',linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()



ridgeRegression = ridge_regression(
    alpha=0.2,
    max_iter=1000,
    random_state=42
)

ridgeRegression.fit(X_train,y_train)


y_ridge_predict = ridgeRegression.predict(X_test)

y_ridge_train_pred = regressor.predict(X_train)
y_ridge_test_pred = regressor.predict(X_test)

r2_score_train_ridge = r2_score(y_train,y_ridge_train_pred)
r2_score_test_ridge = r2_score(y_test,y_ridge_test_pred)

print(f"MSE of train_data {mean_squared_error(y_train,y_ridge_train_pred)}")
print(f"MSE of test data {mean_squared_error(y_test,y_ridge_test_pred)}")
print(f"r2_score of train_data {r2_score_train_ridge}")
print(f"r2_score of test data {r2_score_test_ridge}")


residuals_ridge = y_test-y_ridge_test_pred
plt.scatter(y_ridge_test_pred,residuals_ridge)
plt.axhline(y=0,color='black',linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')     
plt.title('Residual Plot')
plt.show()

#actual vs predicted rigre regression
plt.scatter(y_test,y_ridge_test_pred,alpha=0.5)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()  
