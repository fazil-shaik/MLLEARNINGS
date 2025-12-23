import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Speed (km/h)
X = np.array([20, 40, 60, 80, 100, 120]).reshape(-1, 1)

# Fuel consumption (km/l)
y = np.array([10, 16, 22, 20, 15, 10])


#polynomial check

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

#model
model = LinearRegression()
model.fit(X_poly,y)

#plot
y_predict = model.predict(X_poly)

new_fuel = np.array([[24]])
new_usage_poly = poly.transform(new_fuel)
predicted_usage = model.predict(new_usage_poly)


plt.scatter(X,y,label="Actual data")
plt.plot(X,y_predict,label="fuel usage",color="red")
plt.plot(new_fuel,predicted_usage,marker="*",color="Yellow")
plt.xlabel("Speed of data")
plt.ylabel("Fuel usage by speed")
plt.show()



print("the fuel usage is ",y_predict)
print("the fuel usage is ",predicted_usage)