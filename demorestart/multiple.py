from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error


df = load_diabetes()

X = df.data
y = df.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


print("Original features:", X_train.shape[1])
print("Polynomial features:", X_train_poly.shape[1])



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)



model = Ridge(alpha=10)

model.fit(X_train_scaled, y_train)



y_predict = model.predict(X_test_scaled)



r2 = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)

print("\nModel Performance")
print("-----------------")
print("R2 Score:", r2)
print("MAE:", mae)



test_data = X_test[0].reshape(1, -1)

test_data_poly = poly.transform(test_data)

test_data_scaled = scaler.transform(test_data_poly)

prediction = model.predict(test_data_scaled)

print("\nSingle Prediction")
print("-----------------")
print("Actual:", y_test[0])
print("Predicted:", prediction[0])




plt.scatter(y_test, y_predict)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial Ridge Regression")

# plt.savefig("poly_ridge.png")

plt.show()

for alpha in [0.01, 0.1, 1, 10, 50, 100, 500]:

    model = Ridge(alpha=alpha)

    model.fit(X_train_scaled, y_train)

    prediction = model.predict(X_test_scaled)

    print(
        "alpha:", alpha,
        "R2:", r2_score(y_test, prediction),
        "MAE:", mean_absolute_error(y_test, prediction)
    )


plt.scatter(y_test, y_predict)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial Regression")

plt.savefig("polychecking.png")

plt.show()

