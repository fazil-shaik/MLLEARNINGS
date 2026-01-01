import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Temperature': [25, 28, 32, 35, 38, 40, 42],
    'IceCream_sales': [10, 14, 18, 20, 24, 26, 28]
})

X=df[['Temperature']].values
Y=df['IceCream_sales'].values


#train test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#model fitting
model = LinearRegression()
model.fit(X_train,Y_train)

#prediction 
Y_prediction = model.predict(X)
new_value = 41
y_new_prediction = model.predict([[new_value]])


print("Linear regression")

print(f"Sales:{model.intercept_:.2f} + {model.coef_[0]:.2f} * Temperature")
print(f"predicted result is {y_new_prediction[0]:.3f} for value:{new_value}")

plt.figure(figsize=(10,6))
plt.scatter(X,Y,color="red",label="Actual data")
plt.plot(X,Y_prediction,color="purple",label='Predicted values')
plt.legend()
plt.xlabel("Temperature in 'C ")
plt.ylabel("Sales of Icecreams")
plt.show()