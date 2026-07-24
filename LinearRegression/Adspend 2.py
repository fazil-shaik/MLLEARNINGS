import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Adspent":[10,15,20,25,30,35,40,45,80,55,60],
    "Sales":[100,150,180,220,280,300,382,400,450,455,540]
})

# X and Y
X = df[['Adspent']]
Y = df['Sales']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict test
Y_prediction = model.predict(X_test)

# New value prediction
new_value = 20
new_value_prediction = model.predict([[new_value]])

print(f"New Adspent {new_value} → Predicted Sales: {new_value_prediction[0]:.2f}")

print("\nLinear Regression Equation:")
print(f"Sales = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Adspent")

print("R2 Score:", r2_score(Y_test, Y_prediction))

plt.figure(figsize=(10,6))

# Actual data
plt.scatter(X, Y, color='red', label='Actual data')

# Regression line
plt.plot(X, model.predict(X), color='purple', label='Regression line')

plt.scatter(new_value, new_value_prediction, 
            color='green', s=150, marker='X', 
            label='New predicted value')

plt.xlabel('Ad Spend')
plt.ylabel('Sales')
plt.legend()
plt.show()
