from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error,mean_squared_error,mean_absolute_error as mae
df = pd.read_csv('../salary_300_rows.csv')
#preprocessing done 
df.dropna()

# 1. Prepare the data (Predicting Salary)
features = ['ExperienceYears', 'EducationLevel', 'Certifications', 'SkillScore', 'Age']
X = df[features]
y = df['Salary']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit MLR Model
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# 4. Predict
y_pred = mlr_model.predict(X_test)

#check train vs test error 
train_score = mlr_model.score(X_train, y_train)
test_score = mlr_model.score(X_test, y_test)

print(f"Training Score (R^2): {train_score:.4f}")
print(f"Test Score (R^2): {test_score:.4f}")

# 5. Review Coefficients
coeff_df = pd.DataFrame(mlr_model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
print(f"\nIntercept: {mlr_model.intercept_:.2f}")

#lets predict new values
# Wrap your new data in a DataFrame with the correct column names
# new_df = pd.DataFrame([[18, 4, 4, 148143, 26]], columns=X.columns)
# y_new_prediction = mlr_model.predict(new_df)

# print(f"Predicted Value: {y_new_prediction[0]:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted (MLR)')
plt.show()

print(f"MAE: {mae(y_test, y_pred):.2f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"Model Score (R^2): {mlr_model.score(X_test, y_test):.4f}")

# 1. Correct the input values to match realistic scales
# Experience: 18, Edu: 4, Certs: 4, SkillScore: 90, Age: 26
new_df = pd.DataFrame([[10, 4, 7, 90, 24]], columns=X.columns)
y_new_prediction = mlr_model.predict(new_df)

print(f"Predicted Salary: ${y_new_prediction[0]:,.2f}")

# 2. Correct the Plotting Logic
plt.figure(figsize=(10, 6))

# Use Salary for the Y-axis since that is what you are predicting!
plt.scatter(df['ExperienceYears'], df['Salary'], color='blue', alpha=0.3, label='Actual Data')

# Plot the prediction
plt.scatter(18, y_new_prediction, color='red', marker='*', s=200, label='New Prediction')

plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Experience vs Salary: MLR Prediction')
plt.legend()
plt.show()