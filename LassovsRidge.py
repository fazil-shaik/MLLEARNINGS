from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

X = np.array([
    [800, 2, 1, 10, 1, 0],
    [1000, 3, 2, 8, 0, 1],
    [1200, 3, 2, 6, 1, 0],
    [1500, 4, 3, 4, 0, 1],
    [1800, 4, 3, 2, 1, 0]
])

y = np.array([45, 60, 70, 90, 110])

ridge = make_pipeline(StandardScaler(), Ridge(alpha=1))
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.5))

ridge.fit(X, y)
lasso.fit(X, y)

print("Ridge:", ridge.named_steps['ridge'].coef_)
print("Lasso:", lasso.named_steps['lasso'].coef_)
