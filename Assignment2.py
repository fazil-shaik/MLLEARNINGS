import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
# Loading salary dataset
dataframe = pd.read_csv('./salary.csv')

print(dataframe.head())

# Feature matrix (X) and target vector (y)
X = dataframe[['ExperienceYears', 'EducationLevel', 'SkillScore', 'Age']]
y = dataframe['Salary']

# Test splitting 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction for test data
y_prediction = model.predict(X_test)

# Output prediction and coeff and intercepts
print("Multiple Linear Regression\n")
print("Predicted values:", y_prediction)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


#lets calcuate mean squared error for loss funciton
print(f"mse for MLR is {mean_squared_error(y_prediction,y_test)}")

#lets plot the data
plt.figure()
plt.scatter(y_test, y_prediction)
plt.plot()
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()
#for each and every feature we are plotting prediction and actual plots

features = X.columns

for feature in features:
    plt.figure()
    plt.scatter(X_test[feature], y_test, label="Actual")
    plt.scatter(X_test[feature], y_prediction, label="Predicted")
    
    plt.xlabel(feature)
    plt.ylabel("Salary")
    plt.title(f"Actual vs Predicted Salary vs {feature}")
    plt.legend()
    plt.show()

# New data point
new_data = [[5, 3, 85, 28]]  
# ExperienceYears=5, EducationLevel=3, SkillScore=85, Age=28

prediction = model.predict(new_data)
print("Predicted Salary:", prediction[0])



#Ridge regression model
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)


#Prediction checking 

y_pred = ridge_model.predict(X_test)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Ridge Regression: Actual vs Predicted")
plt.show()
