import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Sample Data
# -----------------------------
# X → Production units
# Y → Production cost
X = np.array([2, 3, 4, 5, 6, 7, 8])
Y = np.array([45, 55, 60, 70, 75, 85, 90])

# -----------------------------
# 2. Reshape Data (IMPORTANT for ML)
# -----------------------------
# sklearn expects 2D array for input
X_reshaped = X.reshape(-1, 1)

# -----------------------------
# 3. Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_reshaped, Y)

# -----------------------------
# 4. Predict New Value
# -----------------------------
new_units = 10
predicted_cost = model.predict(np.array([[new_units]]))

print(f"Predicted production cost for {new_units} units: {predicted_cost[0]:.2f}")

# -----------------------------
# 5. Plotting
# -----------------------------
# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 5))

# (A) Scatter plot - actual data
ax.scatter(X, Y, label="Actual Data")

# (B) Regression line
X_line = np.linspace(X.min(), new_units, 100)
Y_line = model.predict(X_line.reshape(-1, 1))
ax.plot(X_line, Y_line, linestyle='--', label="Regression Line")

# (C) Predicted point
ax.scatter(new_units, predicted_cost, label="Predicted Cost")

# -----------------------------
# 6. Labels, Title, Grid
# -----------------------------
ax.set_title("Production Units vs Production Cost")
ax.set_xlabel("Production Units")
ax.set_ylabel("Production Cost")

ax.legend()
ax.grid(True)

# -----------------------------
# 7. Show Plot
# -----------------------------
plt.show()
