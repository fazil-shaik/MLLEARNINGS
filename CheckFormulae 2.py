import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
#linear models 

#formulae = y=mx+b

# Random generation 
X = np.random.rand(200,3)
y = 4*X[:,0] - 2*X[:,1]+ 3*X[:,2]+np.random.randn(200)*0.2 

#train and test split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection
model = LinearRegression()
model.fit(X_train,y_train)

#prediction and plottting
y_pred = model.predict(X_test)

test_data = model.predict(X_test)
train_data = model.predict(X_train)

print(f"mean square error test data {mean_squared_error(y_test,test_data)}")
print(f"mean square error train data {mean_squared_error(y_train,train_data)}")
print(f" r2 error test data {r2_score(y_test,test_data)}")
print(f"r2 error train data {r2_score(y_train,train_data)}")


print(f"All predictions are {y_pred}")

#plot the data
# plt.figure(figsize=(10,8))
# plt.scatter(y_test,y_pred,label='Orginal data',color='red')
# plt.plot(y_pred,label='predicted data',color='blue',linewidth=2)
# plt.xlabel("Orginal age ")
# plt.ylabel("salary hike")
# plt.legend()
# plt.show()

#metrics got for linear regression
# mean square error test data 0.041588477618473146
# mean square error train data 0.04151308533000987
#  r2 error test data 0.9805077942568056
# r2 error train data 0.9843833636589749

#no overfitting no underfitting found

#lets check for non linear models

#decision tree regression
# Split data into regions

# Prediction:

# 𝑦 = Mean of region

np.random.seed(42)

# 1 Feature for visualization clarity
X = np.sort(5 * np.random.rand(200, 1), axis=0)

# Non-linear relationship
y = np.sin(X).ravel() + np.random.randn(200) * 0.2


#train splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeRegressor(
    max_depth=4,        # control tree size
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

dt.fit(X_train, y_train)

y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))

print("Train R2:", r2_score(y_train, y_train_pred))
print("Test R2:", r2_score(y_test, y_test_pred))

print("Feature Importance:", dt.feature_importances_)

X_grid = np.arange(0, 5, 0.01).reshape(-1, 1)
y_grid_pred = dt.predict(X_grid)

plt.scatter(X, y, color="gray", alpha=0.5)
plt.plot(X_grid, y_grid_pred, color="red")
plt.title("Decision Tree Regression")
plt.show()
