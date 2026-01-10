import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])


# House features: [size (sqft), bedrooms]
X = np.array([
    [800, 2],
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4],
    [2000, 4]
])

# House prices (in lakhs)
y = np.array([40, 50, 65, 80, 95, 110])

ridge_model.fit(X, y)

coefficients = ridge_model.named_steps["ridge"].coef_
intercept = ridge_model.named_steps["ridge"].intercept_

all_predictions = ridge_model.predict(X)


print("Intercept:", intercept)
print("Coefficients:", coefficients)


new_house = np.array([[1600, 3]])
predicted_price = ridge_model.predict(new_house)
print("All weigths preidctions : ",all_predictions)

print("Predicted price:", predicted_price)

sizes = X[:, 0]  # size of house
sorted_idx = np.argsort(sizes)

plt.figure(figsize=(10, 4))
plt.plot(sizes[sorted_idx], y[sorted_idx], marker='o', label='Actual Price')
plt.plot(sizes[sorted_idx], all_predictions[sorted_idx], marker='o', label='Predicted Price')

plt.xlabel("House Size (sqft)")
plt.ylabel("House Price (lakhs)")
plt.title("Ridge Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()