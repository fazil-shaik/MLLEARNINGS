import pandas as pd
import numpy as np

data = pd.read_csv("./train.csv")

print("Data shape:", data.shape)

if data.shape[1] != 785:
    raise ValueError(
        f"Expected 785 columns, but found {data.shape[1]}. "
        "Make sure you are using MNIST train.csv."
    )

data = np.array(data)

m, n = data.shape

np.random.shuffle(data)

data_dev = data[:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:n]

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_dev:", X_dev.shape)
print("Y_dev:", Y_dev.shape)

X_train = X_train / 255.0
X_dev = X_dev / 255.0


def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((10, 1))

    W2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def ReLU_derivative(Z):
    return Z > 0


def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)

    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def one_hot(Y):
    result = np.zeros((10, Y.size))
    result[Y.astype(int), np.arange(Y.size)] = 1
    return result


def backward_prop(Z1, A1, A2, W2, X, Y):
    m = X.shape[1]

    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y

    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T @ dZ2 * ReLU_derivative(Z1)

    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)


def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):

        Z1, A1, Z2, A2 = forward_prop(
            W1,
            b1,
            W2,
            b2,
            X
        )

        dW1, db1, dW2, db2 = backward_prop(
            Z1,
            A1,
            A2,
            W2,
            X,
            Y
        )

        W1, b1, W2, b2 = update_params(
            W1,
            b1,
            W2,
            b2,
            dW1,
            db1,
            dW2,
            db2,
            alpha
        )

        if i % 10 == 0:
            predictions = get_predictions(A2)

            accuracy = get_accuracy(
                predictions,
                Y
            )

            print(
                "Iteration:",
                i,
                "Accuracy:",
                round(accuracy * 100, 2),
                "%"
            )

    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(
    X_train,
    Y_train,
    500,
    0.1
)


_, _, _, A2_dev = forward_prop(
    W1,
    b1,
    W2,
    b2,
    X_dev
)

predictions_dev = get_predictions(A2_dev)

accuracy_dev = get_accuracy(
    predictions_dev,
    Y_dev
)

print()
print("Dev Accuracy:", round(accuracy_dev * 100, 2), "%")

index = 0

print("Prediction:", predictions_dev[index])
print("Actual:", Y_dev[index])