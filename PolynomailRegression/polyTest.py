import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("./cars_price.csv")

print(df.head())

X=df[['Age']]
y=df['Price']

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(X)


model = LinearRegression()
model.fit(x_poly, y)


age = [[6]]
age_poly = poly.transform(age)

predicted_price = model.predict(age_poly)
print("Predicted price:", predicted_price)

plt.scatter(X, y, color='blue')

X_range = np.linspace(0, 10, 100).reshape(-1,1)
X_range_poly = poly.transform(X_range)
y_pred = model.predict(X_range_poly)

plt.plot(X_range, y_pred, color='red')
plt.xlabel("Car Age")
plt.ylabel("Car Price")
plt.show()
