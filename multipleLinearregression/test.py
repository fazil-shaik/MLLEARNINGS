import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#formulae y=mx+b(linear regression) for mlr-y=m1*x1+m2*X2....m'n*x'm+b
#mlr genrally consist of multiple inputs or features to predict specific outcome

df = pd.DataFrame({
   'Size_sqft': [800, 1000, 1200, 1500, 1800, 2000, 2300, 2600, 3000],
    'Bedrooms': [1, 2, 2, 3, 3, 3, 4, 4, 5],
    'Age_years': [20, 15, 10, 8, 5, 6, 4, 3, 2],
    'Distance_km': [12, 10, 8, 7, 6, 5, 4, 3, 2],
    'Parking': [0, 0, 1, 1, 1, 1, 1, 1, 1],
    'Price_lakhs': [40, 55, 75, 95, 120, 135, 160, 185, 220]
})

#test fitting
X = df[['Size_sqft','Bedrooms','Age_years','Distance_km','Parking']]
Y = df['Price_lakhs']

#train test split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


#model selecction and fitting
model = LinearRegression()
model.fit(X_train,y_train)

#prediction
y_prediction = model.predict(X_test)

new_value = [900,3,14,9,1]
new_value_prediction = model.predict([new_value])


print("\n multiple linear regression")
print(f"prediction value is {y_prediction} with coeff as {model.coef_[0]:.0f} and intercept of {model.intercept_:.0f}")
print(f"new value prediction is {new_value_prediction}")

from sklearn.metrics import r2_score
score = r2_score(y_test, y_prediction)
print(f"\nModel Accuracy (R² Score): {score:.2f}")


plt.figure()
plt.scatter(y_test, y_prediction)
plt.xlabel("Actual Price (lakhs)")
plt.ylabel("Predicted Price (lakhs)")
plt.title("Actual vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.show()
