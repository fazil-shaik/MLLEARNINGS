import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# House age in years
X = np.array([1, 5, 10, 15, 20, 30, 40, 50]).reshape(-1, 1)

# Price in lakhs (₹) – non-linear pattern
y = np.array([95, 90, 80, 70, 65, 68, 75, 85])

#set polynomaial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

#model 
model = LinearRegression()
model.fit(X_poly,y)

#prediction
Y_prediction = model.predict(X_poly)

new_age = np.array([[5]])
new_age_poly = poly.transform(new_age)
predicted_price = model.predict(new_age_poly)
print("Predicted price: ",predicted_price)


plt.scatter(X,y)
plt.plot(X,Y_prediction)
plt.scatter(new_age, predicted_price, marker='*', s=100, label="Prediction (Age=25)")
plt.xlabel("House Age (years)")
plt.ylabel("Price (Lakhs)")
plt.title("House Price vs Age (Polynomial Regression)")
plt.show()