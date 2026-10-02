import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

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



model = Pipeline(
    [
        ('scaler',StandardScaler()),
        ('SVR',SVR())
    ]
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

test_r2 = r2_score(
    y_test,
    y_pred
)

train_r2 = r2_score(
    y_train,
    model.predict(X_train)
)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("Test R² :", test_r2)
print("Train R²:", train_r2)
print("Gap :", train_r2 - test_r2)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

for c in [0.01, 0.1, 1, 10, 100]:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(
            C=c
        ))
    ])

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
        f"C={c:<6} | "
        f"Train R²={train_r2:.4f} | "
        f"Test R²={test_r2:.4f} | "
        f"Gap={train_r2-test_r2:.4f}"
    )