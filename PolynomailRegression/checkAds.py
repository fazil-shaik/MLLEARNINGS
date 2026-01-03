import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error


data = {
    "AdSpend": [10, 20, 30, 40, 50, 60, 70, 80],
    "TVAds": [5, 7, 9, 12, 15, 18, 20, 22],
    "OnlineAds": [2, 4, 6, 8, 10, 12, 14, 16],
    "Sales": [100, 150, 210, 280, 360, 430, 510, 600]
}

df = pd.DataFrame(data)
df.to_csv("marketing_sales.csv", index=False)

print("CSV created successfully!\n")


df = pd.read_csv("marketing_sales.csv")
print("Dataset:")
print(df, "\n")


X = df[['AdSpend', 'TVAds', 'OnlineAds']]   # MULTIPLE FEATURES
y = df['Sales']                            # TARGET


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape, "\n")


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

print("Linear Regression Results")
print("Predictions:", y_pred)
print("Actual:", y_test.values)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_, "\n")


new_data = [[45, 14, 9]]  # AdSpend, TVAds, OnlineAds
new_prediction = linear_model.predict(new_data)

print("Prediction for new input:", new_prediction, "\n")


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

poly_model = LinearRegression()
poly_model.fit(X_train_p, y_train_p)

y_poly_pred = poly_model.predict(X_test_p)

print("Polynomial Regression Results")
print("R2 Score:", r2_score(y_test_p, y_poly_pred))
print("MSE:", mean_squared_error(y_test_p, y_poly_pred))
