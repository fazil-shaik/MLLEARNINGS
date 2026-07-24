# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

dataset = pd.read_csv('./HousingPrice.csv')

cols_to_use = [
    'Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname',
    'Propertycount', 'Distance', 'CouncilArea', 'Bedroom2',
    'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price'
]

dataset = dataset[cols_to_use]

# ===============================
# Handle Missing Values
# ===============================
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

dataset['Landsize'] = dataset['Landsize'].fillna(dataset['Landsize'].mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset['BuildingArea'].mean())

# Drop rows where target is missing
dataset.dropna(inplace=True)

# ===============================
# One-Hot Encoding
# ===============================
dataset = pd.get_dummies(dataset, drop_first=True)

# ===============================
# Train-Test Split
# ===============================
X = dataset.drop('Price', axis=1)
y = dataset['Price']

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=2
)

# ===============================
# Linear Regression (Overfitting)
# ===============================
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_X, train_y)

print("Linear Regression Train R2:", lr.score(train_X, train_y))
print("Linear Regression Test R2 :", lr.score(test_X, test_y))

# ===============================
# Lasso Regression (L1)
# ===============================
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=50, max_iter=100, tol=0.1)
lasso.fit(train_X, train_y)

print("\nLasso Regression Train R2:", lasso.score(train_X, train_y))
print("Lasso Regression Test R2 :", lasso.score(test_X, test_y))

# ===============================
# Ridge Regression (L2)
# ===============================
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge.fit(train_X, train_y)

print("\nRidge Regression Train R2:", ridge.score(train_X, train_y))
print("Ridge Regression Test R2 :", ridge.score(test_X, test_y))

# ===============================
# Compare number of features used
# ===============================
print("\nTotal Features:", X.shape[1])
print("Lasso Non-zero Coefficients:", np.sum(lasso.coef_ != 0))
