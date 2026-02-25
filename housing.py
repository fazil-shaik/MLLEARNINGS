import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# X represents House Size
X = 2 * np.random.rand(100, 1)  
# y represents Price (Actual relationship: Price = 3*Size + 4 + noise)
y = 4 + 3 * X + np.random.randn(100, 1) 


learning_rate = 0.1
iterations = 1000
m = len(y) # Total number of houses in our data

# We start with a random guess for our slope (m) and intercept (b)
# theta[0] will be the intercept, theta[1] will be the slope
theta = np.random.randn(2, 1) 

# Math trick: Add a column of 1s to X so we can multiply the intercept easily
X_b = np.c_[np.ones((m, 1)), X]

# Create an empty list so we can watch the error drop over time
cost_history = []


for i in range(iterations):
    # A. Make predictions with our current guess
    prediction = X_b.dot(theta)
    
    # B. Calculate how wrong the predictions are (Error)
    error = prediction - y
    
    # C. Calculate the Gradients (The 'slope' of the error curve)
    # This tells us which direction to adjust our line
    gradients = (2/m) * X_b.T.dot(error)
    
    # D. Update our slope and intercept
    # We subtract because we want to move *down* the error slope
    theta = theta - learning_rate * gradients
    
    # E. Calculate the Mean Squared Error (Cost) for this step and save it
    cost = np.mean(error**2)
    cost_history.append(cost)


print("--- Final Model Parameters ---")
print(f"Intercept (Base Price): {theta[0][0]:.2f}")
print(f"Slope (Price per sq ft): {theta[1][0]:.2f}")

# Plotting the results
plt.figure(figsize=(12, 5))

# Plot 1: The Best Fit Line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Houses')
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)
plt.plot(X_new, y_predict, 'r-', linewidth=2, label='Our Model')
plt.title("House Size vs Price")
plt.xlabel("House Size")
plt.ylabel("Price")
plt.legend()

# Plot 2: The Error Dropping Over Time
plt.subplot(1, 2, 2)
plt.plot(cost_history, color='purple')
plt.title("Error (Cost) Over Time")
plt.xlabel("Step Number (Iteration)")
plt.ylabel("Amount of Error")

plt.tight_layout()
plt.show()


def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        prediction = X.dot(theta)
        error = prediction - y
        gradients = (2/m) * X.T.dot(error)
        theta = theta - learning_rate * gradients
        cost = np.mean(error**2)
        cost_history.append(cost)
    
    return theta, cost_history

# Example usage of the function
theta_final, cost_history_final = gradient_descent(X_b, y, theta, learning_rate, iterations)
print("--- Final Model Parameters from Function ---")
print(f"Intercept (Base Price): {theta_final[0][0]:.2f}")
print(f"Slope (Price per sq ft): {theta_final[1][0]:.2f}")  

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Houses')
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_final)
plt.plot(X_new, y_predict, 'r-', linewidth=2, label='Our Model from Function')
plt.title("House Size vs Price (From Function)")
plt.xlabel("House Size")
plt.ylabel("Price")
plt.legend()    
plt.subplot(1, 2, 2)
plt.plot(cost_history_final, color='orange')
plt.title("Error (Cost) Over Time (From Function)")
plt.xlabel("Step Number (Iteration)")
plt.ylabel("Amount of Error")
plt.tight_layout()
plt.show()