import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


dataframe = pd.read_csv('./delivery_data_1000.csv')
print(dataframe.head())


X = dataframe.drop(columns=['delivery_time_minutes'])
y = dataframe['delivery_time_minutes']

X = pd.get_dummies(X)
#model training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model training

model = DecisionTreeRegressor(
    min_samples_leaf=10,
    max_depth=3,
    min_samples_split=2,
    random_state=42
)

model.fit(X_train,y_train)


#predicitng the model
y_prediction = model.predict(X_test)



new_data = {
    'distance_km': 4.5,
    'time_of_day': 'afternoon',
    'traffic_level': 'medium',
    'weather': 'clear',
    'prep_time_min': 12
}

new_df = pd.DataFrame([new_data])

new_df = pd.get_dummies(new_df)
new_df = new_df.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(new_df)
print("Predicted delivery time:", prediction[0])




# new_value = [['2.5',str("afternoon"),'medium','clear','20']]
# y_new_prediction = model.predict(new_value)

# print(f"new value predicitions are {y_new_prediction}")

#prinitng
print(f"Mean square error is {mean_squared_error(y_test,y_prediction)}")
print(f"All predictions are {y_prediction}")




# Predictions
y_pred = model.predict(X_test)

# Plot: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.title("Actual vs Predicted Delivery Time")
plt.show()

# Plot: Error distribution
errors = y_test - y_pred

plt.figure()
plt.hist(errors, bins=20)
plt.xlabel("Prediction Error (minutes)")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.show()

mean_squared_error(y_test, y_pred)

plt.figure(figsize=(10,6))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True
)
plt.show()

#train error 
y_train_pred = model.predict(X_train)
train_error = mean_squared_error(y_train, y_train_pred)

print("Train Error:", train_error)
#test error

y_test_pred = model.predict(X_test)
test_error = mean_squared_error(y_test, y_test_pred)

print("Test Error:", test_error)
