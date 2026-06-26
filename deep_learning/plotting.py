import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor


X = [
 [0,0],
 [0,1],
 [1,0],
 [1,1]
]

Y = [
 [0],
 [1],
 [1],
 [0]
]

W1 = 2
W2 = 4

b1 = 4
b2 = 2


np.random.seed(42)

# Network architecture
input_size = 2
hidden_size = 4
output_size = 1

# Input → Hidden
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

# Hidden → Output
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

print("W1:\n", W1)
print("\nb1:\n", b1)

print("\nW2:\n", W2)
print("\nb2:\n", b2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def predict(X, W1, W2, b1, b2):
    h1 = sigmoid(W1 * X + b1)
    h2 = relu(W2 * h1 + b2)
    return h2


x1_min, x1_max = -2.5, 2.5
x2_min, x2_max = -2.5, 2.5
xx1, xx2 = np.meshgrid(
    np.arange(x1_min, x1_max, 0.02),
    np.arange(x2_min, x2_max, 0.02)
)

Z = predict(np.c_[xx1.ravel(), xx2.ravel()], W1, W2, b1, b2).reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), s=40, edgecolor='k', cmap=plt.cm.RdYlBu)
plt.title("XOR function approximation")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()




# Easy for quick plots
# plt.figure(figsize=(10, 6))
# plt.scatter([1.1,1.4,2.2,2.223,2.95],[1.4,4,2,5,8.97],colorizer='blue',alpha=0.5)
# plt.plot([1, 2, 3], [1, 4, 9])
# plt.title('Simple Plot')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

# fig, ax = plt.subplots(figsize=(10, 6))command:python.viewOutput
# ax.plot([1, 2, 3], [1, 4, 9])
# ax.set_title('Object-Oriented Plot')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# plt.show()

