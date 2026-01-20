from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('./salary.csv')

X = np.array(df[['ExperienceYears','EducationLevel','SkillScore','Age']]).reshape(-1, 4)
y = np.array(df['Salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

y_pred = lasso_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Coefficients: {lasso_model.coef_}') 
print(f'Intercept: {lasso_model.intercept_}')
print(f'R^2 Score: {lasso_model.score(X_test, y_test)}')


plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Lasso Regression: Actual vs Predicted Salary')
plt.legend()
plt.show()