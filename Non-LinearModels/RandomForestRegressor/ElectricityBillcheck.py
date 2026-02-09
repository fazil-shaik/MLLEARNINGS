import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
n = 1000

temperature = np.random.uniform(10, 40, n)
hour = np.random.randint(0, 24, n)
occupants = np.random.randint(1, 50, n)
is_weekend = np.random.choice([0, 1], n)

# Realistic non-linear energy consumption pattern
energy_consumption = (
    0.6 * temperature +
    0.8 * occupants +
    5 * np.sin(hour / 24 * 2 * np.pi) +
    8 * is_weekend +
    np.random.normal(0, 3, n)
)

data = pd.DataFrame({
    "temperature": temperature,
    "hour": hour,
    "occupants": occupants,
    "is_weekend": is_weekend,
    "energy_kwh": energy_consumption
})

print(data.head())


X = data.drop('energy_kwh',axis=1)
y = data["energy_kwh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=9,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train,y_train)


y_pred = model.predict(X_test)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"rmse is {rmse}")
print(f"r2_score value is {r2}")

print(f"predicted values is {y_pred}")

train_mse = mean_squared_error(y_train,y_train_pred)
test_mse = mean_squared_error(y_test,y_test_pred)


print(f"train mse is {train_mse}")
print(f"test mse is {test_mse}")



new_data = pd.DataFrame({
    "temperature": [32],
    "hour": [15],
    "occupants": [20],
    "is_weekend": [0]
})
prediction = model.predict(new_data)

print(f"new prediction value is {prediction}")
# #plotting 
# import matplotlib.pyplot as plt

# features = X.columns
# plt.figure(figsize=(10,8))
# for i,feature in enumerate(features,1):
#     plt.subplot(2,2,i)
#     plt.scatter(X[feature],y)
#     plt.xlabel(feature)
#     plt.ylabel("Energy (kWh)")
# plt.tight_layout()
# # plt.scatter(X,y,label='Orginal values',color='blue')
# plt.plot(y_test,y_pred,label='Predicted data',color='purple')
# plt.xlabel("Actual Energy Consumption (kWh)")
# plt.ylabel("Predicted Energy Consumption (kWh)")
# plt.title("Random Forest Regression: Actual vs Predicted")
# plt.show()

# residualstest = y_test - y_pred

# plt.figure(figsize=(10, 8))
# for i, feature in enumerate(X_test.columns, 1):
#     plt.subplot(2, 2, i)
#     plt.scatter(X_test[feature], residualstest,color='red')
#     plt.axhline(0,color='blue')
#     plt.xlabel(feature)
#     plt.ylabel("Residual testing data")

# plt.tight_layout()
# plt.show()

# residualstraining = y_train-y_train_pred

# plt.figure(figsize=(10,8))
# for i, feat in enumerate(X_train.columns, 1):
#     plt.subplot(2, 2, i)
#     plt.scatter(X_train[feat], residualstraining)
#     plt.axhline(0)
#     plt.xlabel(feat)
#     plt.ylabel("Training Residuals")

# plt.tight_layout()
# plt.show()


