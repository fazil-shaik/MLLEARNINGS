import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Sample data: Production units (X) vs Production cost (Y)
X = np.array([2, 3, 4, 5, 6, 7, 8])
Y = np.array([45, 55, 60, 70, 75, 85, 90])

#reshape data
hours = X.reshape(-1,1)
Exam = Y

#model traning
model = LinearRegression()
model.fit(hours, Exam)

#predicting values De
new_hours = 4.5
predicted_cost = model.predict(np.array([[new_hours]]))

print(f"predicted production cost for {new_hours} units: {predicted_cost[0].round(2)}")

fig,ax = plt.subplots(figsize=(10,6))

ax.scatter(X,Y,label='Actual data')

X_line = np.linspace(X.min(),new_hours,100).reshape(-1,1)
Y_line = model.predict(X_line)
ax.plot(X_line, Y_line, linestyle='--', label="Regression Line")

# Predicted point
ax.scatter(new_hours, predicted_cost, marker='o', label="Predicted Value")

# Labels & title
ax.set_title("Production Units vs Production Cost")
ax.set_xlabel("Production Units")
ax.set_ylabel("Production Cost")

# Grid & legend
ax.grid(True)
ax.legend()

plt.show()