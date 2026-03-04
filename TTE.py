import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

np.random.seed(42)

X = np.linspace(500, 3000, 50).reshape(-1, 1)
y = 80 * X.flatten() + np.random.normal(0, 20000, 50)

# Tree 1
tree1 = DecisionTreeRegressor(max_depth=2)
tree1.fit(X, y)

pred1 = tree1.predict(X)
residual1 = y - pred1

print("R2 after Tree 1:", r2_score(y, pred1))

# Tree 2
tree2 = DecisionTreeRegressor(max_depth=5)
tree2.fit(X, residual1)

pred2 = tree2.predict(X)
residual2 = y - (pred1 + pred2)

print("R2 after Tree 2:", r2_score(y, pred1 + pred2))

# Tree 3
tree3 = DecisionTreeRegressor(max_depth=6)
tree3.fit(X, residual2)

pred3 = tree3.predict(X)

final_pred = pred1 + pred2 + pred3

print("R2 after Tree 3:", r2_score(y, final_pred))

# Plot
plt.scatter(X, y, label="Actual Price")
plt.plot(X, final_pred, color='green', label="Final Prediction")
plt.legend()
plt.title("Manual Boosting Example")
plt.show()