from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




data = load_diabetes()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="disease_progression"
)

print(X.head())
print(y.head())


#splitting the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#model selection 

model = GradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    random_state=42,
)

model.fit(X_train,y_train)



#prediciton
y_pred = model.predict(X_test)


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("="*50)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

train_r2 = r2_score(
    y_train,
    model.predict(X_train)
)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("Test R² :", r2)
print("Train R²:", train_r2)
print("Gap :", train_r2 - r2)
print("="*50)


print("="*50)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

for n in [10, 25, 50, 100, 150, 200, 300]:

    model = GradientBoostingRegressor(
        n_estimators=n,
        learning_rate=0.1,
        max_depth=3,
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
        f"Trees={n:3} | "
        f"Train R²={train_r2:.4f} | "
        f"Test R²={test_r2:.4f} | "
        f"Gap={train_r2-test_r2:.4f}"
    )

print("="*50)


#learning rate chekcing 
print("="*50)

for lr in [0.01, 0.05, 0.1, 0.2, 0.3]:

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        max_depth=3,
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
        f"Learning Rate={lr:.2f} | "
        f"Train R²={train_r2:.4f} | "
        f"Test R²={test_r2:.4f} | "
        f"Gap={train_r2-test_r2:.4f}"
    )
print("="*50)



#changing n _estimators
print("="*50)
print("Checking the n_estimation with learning rate")
learning_rates = [0.01, 0.05, 0.1]
n_estimators_list = [100, 300, 500]

for lr in learning_rates:

    for n in n_estimators_list:

        model = GradientBoostingRegressor(
            n_estimators=n,
            learning_rate=lr,
            max_depth=3,
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
            f"LR={lr:.2f} | "
            f"Trees={n:3} | "
            f"Train R²={train_r2:.4f} | "
            f"Test R²={test_r2:.4f}"
        )




print("="*50)
print("Maxdepth the rate")
for depth in [1, 2, 3, 4, 5]:

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
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
        f"Depth={depth} | "
        f"Train R²={train_r2:.4f} | "
        f"Test R²={test_r2:.4f} | "
        f"Gap={train_r2-test_r2:.4f}"
    )

print("*"*50)

print("early stopping no change")

#early stopping 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=2,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_r2 = r2_score(
    y_train,
    model.predict(X_train)
)

test_r2 = r2_score(
    y_test,
    y_pred
)

print("Requested trees :", 300)
print("Actual trees    :", model.n_estimators_)
print("Train R²        :", train_r2)
print("Test R²         :", test_r2)
print("Gap             :", train_r2 - test_r2)