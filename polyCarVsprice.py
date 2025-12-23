# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# #car age in years
# X = np.array([0, 1, 2, 3, 5, 7, 10]).reshape(-1, 1)

# # Price in lakhs
# y = np.array([10, 8, 6.5, 5.5, 4.5, 4, 3.5])


# #poly things
# poly = PolynomialFeatures(degree=3)
# X_poly = poly.fit_transform(X)

# #model
# model=LinearRegression()
# model.fit(X_poly,y)

# #predeiction
# y_pred = model.predict(X_poly)


# new_age = np.array([[4]])
# new_age_poly = poly.transform(new_age)

# predicted_price = model.predict(new_age_poly)

# print('predicted price is ',predicted_price.__abs__())
# #plot things
# plt.scatter(X,y)
# plt.plot(X,y_pred)
# plt.plot(new_age,predicted_price,color='red',marker='*')
# plt.xlabel("Car Age (years)")
# plt.ylabel("Price (Lakhs)")
# plt.title("Car Price Depreciation")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Car age in years
X = np.array([0, 1, 2, 3, 5, 7, 10]).reshape(-1, 1)

# Price in lakhs
y = np.array([10, 8, 6.5, 5.5, 4.5, 4, 3.5])

# Polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Model
model = LinearRegression()
model.fit(X_poly, y)

# Prediction for training data
y_pred = model.predict(X_poly)

# New car age prediction
new_age = np.array([[4]])
new_age_poly = poly.transform(new_age)   
predicted_price = model.predict(new_age_poly)

new_age1 = np.array([[5]])
new_age_poly1 = poly.transform(new_age)
predicted_price1 = model.predict(new_age_poly1)

print("predicted price for 1: ",predicted_price1[0])

print("Predicted price:", predicted_price[0])

# Plot
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Polynomial Curve")
plt.scatter(new_age, predicted_price, color='red', marker='*', s=150, label="Prediction (Age=4)")
plt.scatter(new_age1,predicted_price1,color="yellow",marker='*',s=200,label="Prediction (Age=5)")

plt.xlabel("Car Age (years)")
plt.ylabel("Price (Lakhs)")
plt.title("Car Price Depreciation")
plt.legend()
plt.show()
