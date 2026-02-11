from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data = fetch_california_housing()

# print(data.feature_names[0:10])
X = data.data
y = data.target   # Continuous values (house prices)
# print(data.keys())
# print(data.DESCR)

#split and train data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection
Model_decision = DecisionTreeRegressor(
    max_depth=3,
    min_samples_leaf=10,
    min_samples_split=3,
    random_state=42
)
Model_decision.fit(X_train,y_train)

#predicting model
y_prediction = Model_decision.predict(X_test)

print(f"decision Tree predictions are {y_prediction}")

plt.figure(figsize=(10,8))
# plt.scatter(X,y,label='Orginal data')
plt.plot(y_test,y_prediction,label='Predicted data',marker='*')
plt.legend()
plt.show()

plot_tree(Model_decision,filled=True)
plt.show()

new_house = [[8.0,   # MedInc
              20.0,  # HouseAge
              6.0,   # AveRooms
              1.0,   # AveBedrms
              1000.0,# Population
              3.0,   # AveOccup
              34.0,  # Latitude
              -118.0 # Longitude
             ]]

prediction = Model_decision.predict(new_house)

print("Predicted house value (in 100k $):", prediction[0])
print("Predicted house value ($):", prediction[0] * 100000)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_prediction,color='red')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Prices")
plt.show()