# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression


# df = pd.DataFrame({
#     'hours':[1,2,3,4,5,6,7,8],
#     'exam_scores':[20,305,55,70,80,85,87,88] 
# })


# #Prep data
# X = np.array(df['hours']).reshape(-1,1)
# y = np.array(df['exam_scores'])

# #splittng data


# #model training
# model=LinearRegression()
# model.fit(X,y)

# #prediction data
# y_predict = model.predict(X)

# print("\n Linear regression")
# print(f" Aquired_marks = {model.intercept_:.0f}+{model.coef_[0]:.0f} *Score")




# #plotting data

# plt.scatter(X,y,alpha=0.3)
# plt.plot(X,y_predict,color="red",linewidth=3)
# plt.xlabel('Time studies')
# plt.ylabel('Exam score')
# plt.legend()
# plt.show()


# from sklearn.preprocessing import PolynomialFeatures

# # polynomial features
# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)

# # model training
# model_poly = LinearRegression()
# model_poly.fit(X_poly, y)

# # prediction
# y_poly_predict = model_poly.predict(X_poly)

# print("\n Polynomial regression (degree 2)")
# print("Equation: y = b0 + b1*x + b2*x^2")
# print("Intercept:", model_poly.intercept_)
# print("Coefficients:", model_poly.coef_)

# # sort for plotting
# sorted_idx = np.argsort(X.flatten())
# X_sorted = X.flatten()[sorted_idx]
# y_poly_sorted = y_poly_predict[sorted_idx]

# # plotting
# plt.scatter(X, y, alpha=0.3)
# plt.plot(X_sorted, y_poly_sorted, color="green", linewidth=3)
# plt.xlabel('Time studies')
# plt.ylabel('Exam score')
# plt.show()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import r2_score

# # -------------------------
# # Dataset
# # -------------------------
# df = pd.DataFrame({
#     'hours': [1, 2, 3, 4, 5, 6, 7, 8],
#     'exam_scores': [20, 30, 55, 70, 80, 85, 87, 88]
# })

# # -------------------------
# # Prepare data
# # -------------------------
# X = np.array(df['hours']).reshape(-1, 1)
# y = np.array(df['exam_scores'])

# # -------------------------
# # Linear Regression
# # -------------------------
# linear_model = LinearRegression()
# linear_model.fit(X, y)

# y_linear_pred = linear_model.predict(X)

# print("\nLinear Regression Equation")
# print(f"y = {linear_model.intercept_:.2f} + {linear_model.coef_[0]:.2f} * x")

# # -------------------------
# # Polynomial Regression (Degree 2)
# # -------------------------
# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)

# poly_model = LinearRegression()
# poly_model.fit(X_poly, y)

# y_poly_pred = poly_model.predict(X_poly)

# print("\nPolynomial Regression Equation (Degree 2)")
# print(f"Intercept: {poly_model.intercept_}")
# print(f"Coefficients: {poly_model.coef_}")

# # -------------------------
# # Sorting for smooth plotting
# # -------------------------
# sorted_index = np.argsort(X.flatten())
# X_sorted = X.flatten()[sorted_index]
# y_linear_sorted = y_linear_pred[sorted_index]
# y_poly_sorted = y_poly_pred[sorted_index]

# # -------------------------
# # Plotting
# # -------------------------
# plt.figure(figsize=(8, 5))
# plt.scatter(X, y, alpha=0.4, label="Actual Data")
# plt.plot(X_sorted, y_linear_sorted, color="red", linewidth=2, label="Linear Regression")
# plt.plot(X_sorted, y_poly_sorted, color="green", linewidth=2, label="Polynomial Regression (Degree 2)")
# plt.xlabel("Hours Studied")
# plt.ylabel("Exam Score")
# plt.legend()
# plt.grid(True)
# plt.show()

# r2_linear = r2_score(y,y_linear_pred)
# r2_poly = r2_score(y,y_poly_pred)


# print(f"\n Accuracy comparison")
# print(f"Linear regression Accuracy: {r2_linear:.4f}")
# print(f"Polynomial regression Accuracy: {r2_poly:.4f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------------
# Dataset
# -------------------------
df = pd.DataFrame({
    'hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'exam_scores': [20, 305, 55, 70, 80, 85, 87, 88]
})

# -------------------------
# Prepare data
# -------------------------
X = df[['hours']].values
y = df['exam_scores'].values

# -------------------------
# Train / Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# LINEAR REGRESSION
# =========================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_linear_pred = linear_model.predict(X_test)

r2_linear = r2_score(y_test, y_linear_pred)

print("\nLinear Regression")
print(f"Equation: y = {linear_model.intercept_:.2f} + {linear_model.coef_[0]:.2f}xMarks")
print(f"R² Score: {r2_linear:.4f}")

# =========================
# POLYNOMIAL REGRESSION (Degree 2)
# =========================
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_poly_pred = poly_model.predict(X_test_poly)

r2_poly = r2_score(y_test, y_poly_pred)

print("\nPolynomial Regression (Degree 2)")
print(f"Intercept: {poly_model.intercept_}")
print(f"Coefficients: {poly_model.coef_}")
print(f"R² Score: {r2_poly:.4f}")

# =========================
# PLOTTING (FULL DATA)
# =========================
# Predict on full dataset for visualization
X_full_sorted = np.sort(X.flatten())

# Linear line
y_linear_line = linear_model.predict(X_full_sorted.reshape(-1, 1))

# Polynomial curve
X_full_poly = poly.transform(X_full_sorted.reshape(-1, 1))
y_poly_line = poly_model.predict(X_full_poly)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.4, label="Actual Data")
plt.plot(X_full_sorted, y_linear_line, color="red", label="Linear Regression")
plt.plot(X_full_sorted, y_poly_line, color="green", label="Polynomial Regression (Degree 2)")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# FINAL COMPARISON
# =========================
print("\nModel Comparison")
print(f"Linear Regression R² : {r2_linear:.4f}")
print(f"Polynomial Regression R² : {r2_poly:.4f}")
