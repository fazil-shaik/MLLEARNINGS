import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,     # number of trees
    max_depth=None,      # grow fully
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(X_range)




# Error
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

#plotting the data
plt.figure()
plt.scatter(X, y,color='blue',label='Actual data')
plt.scatter(X_train,y_train,color='grey',label='training data')
plt.scatter(X_test,y_test,color='yellow',label='Testing data')
plt.plot(X_range, y_range_pred,color='red',label='Predicted value',marker='*')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Random Forest Regression")
plt.show()