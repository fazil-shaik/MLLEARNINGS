#non linear models 

#Decision trees


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



housing = fetch_california_housing()

X = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

y = pd.Series(
    housing.target,
    name="MedHouseValue"
)

print(X.head())
print(y.head())



#model splitting

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


#model training 

model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

model.fit(X_train,y_train)


#model prediction

model_prediction = model.predict(X_test)



#model eval

mae = mean_absolute_error(
    y_test,
    model_prediction
)
print("MAE:", mae)


mse = mean_squared_error(
    y_test,
    model_prediction
)

print("MSE:", mse)


rmse = np.sqrt(mse)

print("RMSE:", rmse)

r2 = r2_score(
    y_test,
    model_prediction
)

print("R²:", r2)
new_house = [
    5.446,
    23.56,
    6.98129,
    1.09938,
    567.8,
    2.8991,
    34.77,
    -119.88
]
y_newPredict = model.predict([new_house])

print("new prediction is ",y_newPredict)


actual = 2.90
predicted = y_newPredict[0]

error = abs(actual - y_newPredict)

print("Actual:", actual)
print("Predicted:", y_newPredict)
print("Absolute error:", error)


percentage_error = (
    abs(actual - y_newPredict) / actual
) * 100

print(
    "Percentage error:",
    percentage_error,
    "%"
)





from sklearn.metrics import r2_score

# Predictions on training data
train_predictions = model.predict(X_train)

# Predictions on test data
test_predictions = model.predict(X_test)

# R² scores
train_r2 = r2_score(
    y_train,
    train_predictions
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

print("Train R²:", train_r2)
print("Test R² :", test_r2)


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

for depth in [2, 3, 5, 8, 10, 15, 20, None]:

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_r2 = r2_score(
        y_train,
        model.predict(X_train)
    )

    test_r2 = r2_score(
        y_test,
        model.predict(X_test)
    )

    print(
        f"Depth={str(depth):>4} | "
        f"Train R²={train_r2:.4f} | "
        f"Test R²={test_r2:.4f} | "
        f"Gap={train_r2-test_r2:.4f}"
    )

# depths = [1, 2, 3, 5, 8, 10, 15, 20]

# for depth in depths:

#     model = DecisionTreeRegressor(
#         max_depth=depth,
#         random_state=42
#     )

#     model.fit(X_train, y_train)

#     train_pred = model.predict(X_train)
#     test_pred = model.predict(X_test)

#     train_r2 = r2_score(
#         y_train,
#         train_pred
#     )

#     test_r2 = r2_score(
#         y_test,
#         test_pred
#     )

#     print(
#         f"Depth={depth:2} | "
#         f"Train R2={train_r2:.4f} | "
#         f"Test R2={test_r2:.4f}"
#     )


