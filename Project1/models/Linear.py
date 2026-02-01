from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error,mean_squared_error,mean_absolute_error as mae
df = pd.read_csv('../../salary.csv')
#preprocessing done 
df.dropna()


# 1. Select only ONE feature for X
X = df[['ExperienceYears']] # Note the double brackets to keep it a 2D array
y = df['Salary']

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit the model
simple_model = LinearRegression()
simple_model.fit(X_train, y_train)

# 4. Predict
y_pred = simple_model.predict(X_test)

# 5. Visualize the "Linear" part
plt.scatter(X_test, y_test, color='gray', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary (Simple Linear Regression)')
plt.legend()
plt.show()

print(f"MAE: {mae(y_test, y_pred):.2f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"Model Score (R^2): {simple_model.score(X_test, y_test):.4f}")