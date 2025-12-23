import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Size': [1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [3, 3, 4, 4, 5],
    'Age': [10, 15, 20, 5, 8],
    'Price': [300000, 350000, 420000, 500000, 600000]
}

df = pd.DataFrame(data)

X=df[['Size','Age','Bedrooms']]
Y=df["Price"]

#train data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=42)

#train mode
model = LinearRegression()
model.fit(X_train,Y_train)

#predict model
y_predict = model.predict(X_test)

mse=mean_squared_error(Y_test,y_predict)
print("MSE:", mse)

r2 = r2_score(Y_test, y_predict)
print("R2 Score:", r2)


coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coefficients)
print("Intercept:", model.intercept_)








