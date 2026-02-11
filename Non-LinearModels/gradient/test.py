import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([3, 5, 7, 10, 12])

#train the model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X,y)

#predict the model 
y_pred = model.predict(X)

#plot predictions
new_values = [[10]]
y_new_value_pred = model.predict(new_values)

print(f"new value prediction is {y_new_value_pred}")

print(f"All predictinos we got is {y_pred}")
print(f"mean square error we have is {mean_squared_error(y,y_pred)}")
print(f"r2_score error is {r2_score(y,y_pred)}")
X_range = np.linspace(X.min(),X.max(),100).reshape(-1,1)
y_range_pred = model.predict(X_range)

plt.scatter(X, y, label="Actual data")
plt.plot(X_range, y_range_pred, label="Gradient Boosting Prediction")
plt.plot(new_values,y_new_value_pred,label='New value prediciton',marker='*')
plt.legend()
plt.show()