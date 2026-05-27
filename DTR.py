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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
# Create dataset
dataset_dict = {
   'Outlook': ['sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain', 'overcast', 
               'sunny', 'sunny', 'rain', 'sunny', 'overcast', 'overcast', 'rain',
               'sunny', 'overcast', 'rain', 'sunny', 'sunny', 'rain', 'overcast',
               'rain', 'sunny', 'overcast', 'sunny', 'overcast', 'rain', 'overcast'],
   'Temp.': [85.0, 80.0, 83.0, 70.0, 68.0, 65.0, 64.0, 72.0, 69.0, 75.0, 75.0,
             72.0, 81.0, 71.0, 81.0, 74.0, 76.0, 78.0, 82.0, 67.0, 85.0, 73.0,
             88.0, 77.0, 79.0, 80.0, 66.0, 84.0],
   'Humid.': [85.0, 90.0, 78.0, 96.0, 80.0, 70.0, 65.0, 95.0, 70.0, 80.0, 70.0,
              90.0, 75.0, 80.0, 88.0, 92.0, 85.0, 75.0, 92.0, 90.0, 85.0, 88.0,
              65.0, 70.0, 60.0, 95.0, 70.0, 78.0],
   'Wind': [False, True, False, False, False, True, True, False, False, False, True,
            True, False, True, True, False, False, True, False, True, True, False,
            True, False, False, True, False, False],
   'Num_Players': [52, 39, 43, 37, 28, 19, 43, 47, 56, 33, 49, 23, 42, 13, 33, 29,
                   25, 51, 41, 14, 34, 29, 49, 36, 57, 21, 23, 41]
}

# Prepare data
df = pd.DataFrame(dataset_dict)
df = pd.get_dummies(df, columns=['Outlook'], prefix='', prefix_sep='')
df['Wind'] = df['Wind'].astype(int)

# Split features and target
X, y = df.drop('Num_Players', axis=1), df['Num_Players']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, shuffle=False)



clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42   )

clf.fit(X_train, y_train)

plt.figure(figsize=(11, 20), dpi=300)

for i, tree_idx in enumerate([0, 2, 24, 49]):
    plt.subplot(4, 1, i+1)
    plot_tree(clf.estimators_[tree_idx,0], 
              feature_names=X_train.columns,
              impurity=False,
              filled=True, 
              rounded=True,
              precision=2,
              fontsize=12)
    plt.title(f'Tree {tree_idx + 1}')

plt.suptitle('Decision Trees from GradientBoosting', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

y_pred = clf.predict(X_test)

results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(results_df) # Display results DataFrame

# Calculate and display RMSE
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_test, y_pred)
print(f"nModel Accuracy: {rmse:.4f}")
