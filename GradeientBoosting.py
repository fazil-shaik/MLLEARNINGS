import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.rand(100,2)*10
y = 4*X+3 + np.random.randn(100,2)*5

w=0
b=0
learning_rate = 0.1
epochs = 100
n = len(X)


for i in range(epochs):
    
    # Predictions
    y_pred = w * X + b
    
    # Compute gradients
    dw = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Print loss every 10 iterations
    if i % 10 == 0:
        loss = np.mean((y - y_pred) ** 2)
        print(f"Epoch {i}, Loss: {loss:.4f}")


print("Final weight:", w)
print("Final bias:", b)

plt.scatter(X, y)
plt.plot(X, w*X + b, color='red')
plt.title("Gradient Descent Linear Regression")
plt.show()
