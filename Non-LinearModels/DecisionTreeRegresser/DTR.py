import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = {
    "trained":    [0, 0, 0, 1, 1, 1, 1],
    "experience": [0, 1, 2, 1, 3, 5, 7],
    "survival_days": [10, 15, 20, 40, 55, 65, 75]
}

df = pd.DataFrame(data)
print(df)

#add preprocessor data
X = df[['trained','experience']]
y = df['survival_days']


#split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#model selection
model = DecisionTreeRegressor(
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X,y)

#predictions 
y_predict = model.predict(X_test)

print("Predictions:", y_predict)
print("Actual:", y_test.values)

#mse calcualtion
mse = mean_squared_error(y_test, y_predict)
print("MSE:", mse)
#new prediction
new_soldier = [[1, 4]]
prediction = model.predict(new_soldier)

print("Expected survival days:", prediction[0])

#plotting the data
plt.figure(figsize=(12,6))
plot_tree(
    model,
    feature_names=["trained", "experience"],
    filled=True
)
plt.show()

#new model prediction with depth of 3
model = DecisionTreeRegressor(max_depth=3)
model.fit(X, y)

model.predict([[1, 4]])