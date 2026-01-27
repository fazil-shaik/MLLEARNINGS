import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

n_samples = 250

years_experience = np.random.uniform(0, 20, n_samples)

salary = (
    25000                                  # base salary
    + 1500 * years_experience              # linear growth
    + 150 * (years_experience ** 2)        # quadratic growth (non-linear)
    + 5000 * np.sin(years_experience / 2)  # market / role fluctuations
)

noise = np.random.normal(0, 1500, n_samples)
salary += noise


X = years_experience.reshape(-1, 1)
y = salary

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#linear regression model
Linear_model = LinearRegression()
Linear_model.fit(X_train,y_train)

#predicing the values 
y_pred = Linear_model.predict(X_test)

#evaluatinf model performance
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
# Visualizing the results
plt.scatter(X,y,color='blue',label='Data Points')
plt.plot(X_test,y_pred,color='red',linewidth=2,label='Linear Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')
plt.legend()
plt.show()


#polynomial regression model
polyNomail = PolynomialFeatures(degree = 3)
x_poly = polyNomail.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(x_poly,y_train)

x_test_poly = polyNomail.transform(X_test)
y_poly_pred = poly_model.predict(x_test_poly)

mse_poly = mean_squared_error(y_test,y_poly_pred)
r2_poly = r2_score(y_test,y_poly_pred)

print(f"Polynomial Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial R^2 Score: {r2_poly:.2f}")
# Visualizing the polynomial regression results
# below is the orginal data points scattering 
plt.scatter(X,y,color='blue',label='Data Points')
# Sorting for a better curve representation
# for not linear line we are sorting the x values of test data to get a smooth curve
sorted_indices = X_test.flatten().argsort()
plt.plot(X_test[sorted_indices],y_poly_pred[sorted_indices],color='green',linewidth=2,label='Polynomial Regression Curve')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Polynomial Regression: Salary vs Years of Experience with degree 3')
plt.legend()
plt.show()  

#polynomial regression model with degree 2
polyNomail = PolynomialFeatures(degree = 2)
x_poly = polyNomail.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(x_poly,y_train)

x_test_poly = polyNomail.transform(X_test)
y_poly_pred = poly_model.predict(x_test_poly)

mse_poly = mean_squared_error(y_test,y_poly_pred)
r2_poly = r2_score(y_test,y_poly_pred)

print(f"Polynomial Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial R^2 Score: {r2_poly:.2f}")
# Visualizing the polynomial regression results
# below is the orginal data points scattering 
plt.scatter(X,y,color='blue',label='Data Points')
# Sorting for a better curve representation
# for not linear line we are sorting the x values of test data to get a smooth curve
sorted_indices = X_test.flatten().argsort()
plt.plot(X_test[sorted_indices],y_poly_pred[sorted_indices],color='green',linewidth=2,label='Polynomial Regression Curve')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Polynomial Regression: Salary vs Years of Experience with degree 2')
plt.legend()
plt.show()  


#polynomial regression model with degree 4
polyNomail = PolynomialFeatures(degree = 4)
x_poly = polyNomail.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(x_poly,y_train)

x_test_poly = polyNomail.transform(X_test)
y_poly_pred = poly_model.predict(x_test_poly)

mse_poly = mean_squared_error(y_test,y_poly_pred)
r2_poly = r2_score(y_test,y_poly_pred)

print(f"Polynomial Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial R^2 Score: {r2_poly:.2f}")
# Visualizing the polynomial regression results
# below is the orginal data points scattering 
plt.scatter(X,y,color='blue',label='Data Points')
# Sorting for a better curve representation
# for not linear line we are sorting the x values of test data to get a smooth curve
sorted_indices = X_test.flatten().argsort()
plt.plot(X_test[sorted_indices],y_poly_pred[sorted_indices],color='green',linewidth=2,label='Polynomial Regression Curve')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Polynomial Regression: Salary vs Years of Experience degree 4')
plt.legend()
plt.show()  

#polynomial regression model with degree 1
polyNomail = PolynomialFeatures(degree = 1 )
x_poly = polyNomail.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(x_poly,y_train)

x_test_poly = polyNomail.transform(X_test)
y_poly_pred = poly_model.predict(x_test_poly)

mse_poly = mean_squared_error(y_test,y_poly_pred)
r2_poly = r2_score(y_test,y_poly_pred)

print(f"Polynomial Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial R^2 Score: {r2_poly:.2f}")
# Visualizing the polynomial regression results
# below is the orginal data points scattering 
plt.scatter(X,y,color='blue',label='Data Points')
# Sorting for a better curve representation
# for not linear line we are sorting the x values of test data to get a smooth curve
sorted_indices = X_test.flatten().argsort()
plt.plot(X_test[sorted_indices],y_poly_pred[sorted_indices],color='green',linewidth=2,label='Polynomial Regression Curve')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Polynomial Regression: underfitted with degree 1')
plt.legend()
plt.show()  

#Overfitting model in polynomial regression

polyNomail = PolynomialFeatures(degree = 1 )
x_poly = polyNomail.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(x_poly,y_train)

x_test_poly = polyNomail.transform(X_test)
y_poly_pred = poly_model.predict(x_test_poly)

mse_poly = mean_squared_error(y_test,y_poly_pred)
r2_poly = r2_score(y_test,y_poly_pred)

print(f"Polynomial Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial R^2 Score: {r2_poly:.2f}")
# Visualizing the polynomial regression results
# below is the orginal data points scattering 
plt.scatter(X,y,color='blue',label='Data Points')
# Sorting for a better curve representation
# for not linear line we are sorting the x values of test data to get a smooth curve
sorted_indices = X_test.flatten().argsort()
plt.plot(X_test[sorted_indices],y_poly_pred[sorted_indices],color='green',linewidth=2,label='Polynomial Regression Curve')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Polynomial Regression: overfitted model degree 10')
plt.legend()
plt.show()  

#bias varinace tradeoff check
degrees = np.arange(1, 15)
train_errors = []
test_errors = []

for degree in degrees:
    # Build the model
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    model = LinearRegression().fit(X_poly_train, y_train)
    
    # Calculate Mean Squared Error
    train_errors.append(mean_squared_error(y_train, model.predict(X_poly_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_poly_test)))

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label='Train Error (Bias)', color='blue', marker='o')
plt.plot(degrees, test_errors, label='Test Error (Variance)', color='red', marker='o')
plt.yscale('log') # Log scale helps see the explosion in variance
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Mean Squared Error')
plt.title('The Bias-Variance Tradeoff')
plt.axvline(x=4, linestyle='--', color='green', label='Sweet Spot (Optimal Complexity)')
plt.legend()
plt.show()