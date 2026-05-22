from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

#load dataset
dataset = pd.read_csv('insurance.csv')

print(dataset.head())

result = dataset.isnull().sum()
print(result)

sns.scatterplot(
    x=dataset['age'],
    y=dataset['charges'],
    hue=dataset['bmi']
)
plt.show()

X = dataset.drop(columns=['charges', 'region'])
y = dataset['charges']

X['sex'] = X['sex'].map({
    'female': 1,
    'male': 0
})
X['smoker'] = X['smoker'].map({
    'yes': 1,
    'no': 0
})

print(X.head())

#train testing splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model selection 
LinearModel = LinearRegression()
LinearModel.fit(X_train, y_train)

#prediction
y_pred = LinearModel.predict(X_test)

#model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(x=y_test, y=y_pred, ax=ax[0])
ax[0].set_title('Actual vs Predicted')
ax[0].set_xlabel('Actual Charges')
ax[0].set_ylabel('Predicted Charges')
sns.histplot(y_test - y_pred, ax=ax[1], kde=True)
ax[1].set_title('Residuals Distribution')
ax[1].set_xlabel('Residuals')
plt.tight_layout()
plt.show()