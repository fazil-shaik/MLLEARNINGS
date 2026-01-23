# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score,mean_absolute_error
# from sklearn.linear_model import LinearRegression,Lasso,Ridge
# import matplotlib.pyplot as plt


# X = np.array([1,2,3,4,5]).reshape(-1,1)
# y = np.array([2,4,6,8,10])

# print(X)

# model = Lasso(alpha=0.3)
# model.fit(X,y)

# print("Coefficient:", model.coef_)
# print(f"Intercept: {model.intercept_:.2f}" )



import numpy as np
from sklearn.linear_model import Lasso

np.random.seed(42)

X = np.random.rand(100, 5)
true_weights = np.array([5, 0, 3, 0, 0])
y = X @ true_weights + np.random.randn(100) * 0.1

lasso_model = Lasso(alpha=0.02)
lasso_model.fit(X,y)

print(f"Lasso coefficients:, {lasso_model.coef_}")