from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Features: area, bedrooms, bathrooms, distance, wall_color, door_type
X = np.array([
    [800, 2, 1, 10, 1, 0],
    [1000, 3, 2, 8, 0, 1],
    [1200, 3, 2, 6, 1, 0],
    [1500, 4, 3, 4, 0, 1],
    [1800, 4, 3, 2, 1, 0]
])

# House price (in lakhs)
y = np.array([45, 60, 70, 90, 110])

model = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.5)
)

model.fit(X, y)

lasso = model.named_steps['lasso']
print("Lasso coefficients:", lasso.coef_)
