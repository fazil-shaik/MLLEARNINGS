import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

X = np.array([
    [500, 1],
    [800, 2],
    [1000, 2],
    [1200, 3],
    [1500, 4]
])

y = np.array([20, 32, 40, 48, 60])

LinearModel = LinearRegression()
LinearModel.fit(X,y)
y_lr = LinearModel.predict(X)


ridge = Ridge(alpha=10)
ridge.fit(X,y)
y_ridge = ridge.predict(X)

new_house = [[1100, 3]]
predicted_price = ridge.predict(new_house)
print(predicted_price)


plt.figure()
plt.plot(y, y, marker='o')
plt.plot(y, y_lr, marker='o')
plt.plot(y, y_ridge, marker='o')
plt.scatter(predicted_price, predicted_price,marker="*")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Price Prediction: Linear vs Ridge Regression")
plt.show()              

