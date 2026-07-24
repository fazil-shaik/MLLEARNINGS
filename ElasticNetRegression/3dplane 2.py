import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mpl_toolkits.mplot3d import Axes3D

URL = "https://storage.googleapis.com/kagglesdsdata/datasets/4312217/7413268/student-mat.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260124%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260124T034812Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1ab85ca5959a7b8679e6483ddcb2cecc2418e3c297aca9ecaed33eceb97cb3b79f5e703b1885925d7d843bbf6f106b172fee0b6ed356b2e36fd1cd5e73402c5573ced2677a4e3876b9adc957bd27d8ae5df1cf6b4b50e7efda52d1ff30e9177f1d67181400e062807662c8ce4949a6885913a2abfb7d755d067ddf0c39d5c116e578f067192a8a7d2364c60cd5cff1839f0e6aa77f59310ad445334b9b39bcd614ce4953c2816dea671102fd373f22cb02582cf20ce0d5834d6d4c689fcd4eb005e3ec99e941004903853e682b0fc52c4543483913e22f83e09294d5ec4613ec1e76f44b9b9d7252491df82ef54bdb94deb43963af1224e0df08c6ebad5798a1"
data = pd.read_csv(URL)

# Preview
print(data[['G1', 'G2', 'G3']].head())

X = data[['G1', 'G2']]
y = data['G3']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error : {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score           : {r2:.2f}")

print("\nModel Coefficients:")
print(f"G1 coefficient: {model.coef_[0]:.3f}")
print(f"G2 coefficient: {model.coef_[1]:.3f}")
print(f"Intercept     : {model.intercept_:.3f}")


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter actual data
ax.scatter(
    X_train['G1'],
    X_train['G2'],
    y_train,
    alpha=0.6,
    label="Actual Data"
)

# Create mesh grid
g1_range = np.linspace(X['G1'].min(), X['G1'].max(), 20)
g2_range = np.linspace(X['G2'].min(), X['G2'].max(), 20)
g1_grid, g2_grid = np.meshgrid(g1_range, g2_range)

# Regression plane equation
g3_plane = (
    model.coef_[0] * g1_grid +
    model.coef_[1] * g2_grid +
    model.intercept_
)

# Plot regression plane
ax.plot_surface(
    g1_grid,
    g2_grid,
    g3_plane,
    alpha=0.4
)

# Labels
ax.set_xlabel("G1 (First Period Grade)")
ax.set_ylabel("G2 (Second Period Grade)")
ax.set_zlabel("G3 (Final Grade)")
ax.set_title("ElasticNet Regression Plane (3D)")

plt.legend()
plt.show()

print("\n ElasticNet 3D regression visualization completed.")
