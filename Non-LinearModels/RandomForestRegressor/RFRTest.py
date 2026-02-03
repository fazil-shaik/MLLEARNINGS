import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Dataset
data = {
    "ExperienceYears": [0,1,2,3,4,5,6],
    "EducationLevel": [1,1,1,1,2,2,2],
    "SkillScore": [45,50,55,60,65,70,72],
    "Age": [21,22,23,24,25,26,27],
    "Salary": [18000,22000,26000,30000,38000,45000,48000]
}

df = pd.DataFrame(data)

X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Plot: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Random Forest Regression: Actual vs Predicted")
plt.show()
