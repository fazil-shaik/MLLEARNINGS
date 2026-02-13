import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.sort(5 * np.random.rand(100,1),axis=0)
y = np.sin(X).ravel()
y+= 0.5 * np.random.rand(y.shape[0])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rd_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
    min_samples_split=5
)


rd_model.fit(X_train,y_train)

train_prediction = rd_model.predict(X_train)
test_prediction = rd_model.predict(X_test)

print("Mean Squared Error (Train):", mean_squared_error(y_train, train_prediction))
print("Mean Squared Error (Test):", mean_squared_error(y_test, test_prediction)) 
print("R^2 Score (Train):", r2_score(y_train, train_prediction))
print("R^2 Score (Test):", r2_score(y_test, test_prediction))


X_test_pred = rd_model.predict(X_test)

X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_pred = rd_model.predict(X_test)


print("shape of X_test:", X_test.shape)
print("shape of y_pred:", y_pred.shape)




# plt.scatter(X, y, label="Actual Data")
# plt.plot(X_test, y_pred, label="Random Forest Prediction")
# plt.legend()
# plt.show()

# new_x = np.array([[2.5]])
# new_y = rd_model.predict(new_x)


# plt.scatter(X, y, label="Training Data")
# plt.plot(X_test, y_pred, label="Random Forest Prediction")
# plt.scatter(new_x, new_y, marker="x", s=100, label="New Prediction")
# plt.legend()
# plt.show()


# plt.style.use('ggplot')
# plt.figure(figsize=(10,8))
# plot_tree(rd_model.estimators_[0],filled=True)
# plt.show()