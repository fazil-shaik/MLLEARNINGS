import numpy as np

X = np.array([
    [1500, 3, 10],
    [1800, 3, 15],
    [2400, 4, 20],
    [3000, 4, 5],
    [3500, 5, 8]
], dtype=float)

y = np.array([300000, 350000, 420000, 500000, 600000], dtype=float).reshape(-1, 1)

m = X.shape[0]  # number of samples

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

X_norm, X_mean, X_std = normalize_features(X)


X_bias = np.hstack((np.ones((m, 1)), X_norm))

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X @ theta
        gradients = (1 / m) * X.T @ (predictions - y)
        theta = theta - alpha * gradients
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


theta = np.zeros((X_bias.shape[1], 1))  # initialize weights
alpha = 0.01
iterations = 1000

theta, cost_history = gradient_descent(
    X_bias, y, theta, alpha, iterations
)

print("Final Parameters (Theta):")
print(theta)

print("\nInitial Cost:", cost_history[0])
print("Final Cost:", cost_history[-1])


def predict(X_new, theta, mean, std):
    X_new_norm = (X_new - mean) / std
    X_new_bias = np.hstack((np.ones((X_new_norm.shape[0], 1)), X_new_norm))
    return X_new_bias @ theta


new_house = np.array([[2000, 3, 10]], dtype=float)
predicted_price = predict(new_house, theta, X_mean, X_std)

print("\nPredicted Price:", predicted_price[0][0])


def r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

y_pred_train = X_bias @ theta
r2 = r2_score(y, y_pred_train)

print("\nR² Score:", r2)
