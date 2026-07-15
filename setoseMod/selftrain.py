#relu function 

def relu(x):
    return max(0,x)

print(relu(5))



#neural network follows two propgations (forward ----> and backward<-------)

# Forward propgation:input layer to layer by layer till we got output


#Activation function

# Hidden = Activation(XW + b)

# Output = Activation(HiddenW + b)


#Backward propagation:it will go layer by layer and predicts calcualtes error,find weights causes error update weights

#New Weight = Old Weight − Learning Rate × Gradient



#full flow 
# Input Data
#      ↓
# Multiply by weights
#      ↓
# Add bias
#      ↓
# Activation
#      ↓
# Prediction
#      ↓
# Calculate loss
#      ↓
# Backpropagation
#      ↓
# Update weights
#      ↓
# Repeat


#first nn single neuron 

import numpy as np

# Inputs
X = np.array([2, 3, 5, 7 ,9, 10])

# Weights
W = np.array([0.5, 0.2,0.4,0.6,0.7,0.9])

# Bias
b = 0.1

# Weighted sum
z = np.dot(X, W) + b

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

output = sigmoid(z)

print("Output:", output)


#2 features and 2 hidden layers


import numpy as np

# Input layer (2 features)
X = np.array([1, 2])

# Hidden layer weights
W1 = np.array([
    [0.2, 0.8],
    [0.5, 0.1]
])

b1 = np.array([0.1, 0.2])

# Output layer weights
W2 = np.array([
    [0.4],
    [0.7]
])

b2 = np.array([0.3])


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Hidden layer
hidden = relu(np.dot(X, W1) + b1)

# Output layer
output = sigmoid(np.dot(hidden, W2) + b2)

print("Hidden:", hidden)
print("Prediction:", output)




#neural network for finding XOR

import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(42)

W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


learning_rate = 0.1


for epoch in range(10000):

    # Forward pass

    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    predictions = sigmoid(z2)

    # Loss

    loss = np.mean((y - predictions) ** 2)

    # Backpropagation

    output_error = y - predictions
    d_output = output_error * sigmoid_derivative(predictions)

    hidden_error = d_output.dot(W2.T)
    d_hidden = hidden_error * sigmoid_derivative(a1)

    # Update weights

    W2 += a1.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


print("\nPredictions:")
print(predictions)