import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('./housepredictions.csv')

print(data.head())

X = data.drop("price",axis=1)
y = data["price"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)


rf = RandomForestRegressor(
    n_estimators=600,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

rf.fit(X_train,y_train)



#prediiction of data points

new_house = [[1200, 3, 2, 8, 5, 10,20]] 
rf.predict(new_house)


y_pred = rf.predict(X_test)


train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

train_err= mean_squared_error(y_train,train_pred)
test_err= mean_squared_error(y_test,test_pred)

print(f"Train Error: {train_err:,.0f}")
print(f"Test Error: {test_err:,.0f}")


# print(f"all predictions {y_pred[5]}")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
