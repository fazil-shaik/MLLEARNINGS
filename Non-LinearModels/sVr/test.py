import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(42)
n = 200

X = np.sort(500 * np.random.rand(n, 1), axis=0)

y = (
    20000 +
    150 * X[:, 0] -
    0.05 * (X[:, 0] ** 2) +
    np.random.randn(n) * 5000
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc_X = StandardScaler()
sc_y = StandardScaler()

X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)

y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).ravel()


svr = SVR(
    kernel='rbf',   # Radial Basis Function (non-linear)
    C=100,
    epsilon=0.1,
    gamma='scale'
)

svr.fit(X_train_scaled, y_train_scaled)


y_pred_scaled = svr.predict(X_test_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1,1))


print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


X_grid = np.linspace(0, 500, 500).reshape(-1,1)
X_grid_scaled = sc_X.transform(X_grid)

y_grid_scaled = svr.predict(X_grid_scaled)
y_grid = sc_y.inverse_transform(y_grid_scaled.reshape(-1,1))

plt.scatter(X, y, alpha=0.4)
plt.plot(X_grid, y_grid, linewidth=2)
plt.xlabel("House Area (sqm)")
plt.ylabel("House Price")
plt.title("Support Vector Regression (RBF Kernel)")
plt.show()
