# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# # Load dataset
# data = fetch_california_housing()

# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = data.target  # Median house value

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Create model
# model = GradientBoostingRegressor(
#     n_estimators=200,
#     learning_rate=0.05,
#     max_depth=3,
#     random_state=42
# )

# # Train model
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluation
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print("Model Performance:")
# print("RMSE:", rmse)
# print("R² Score:", r2)

# # Feature Importance
# importance = pd.Series(model.feature_importances_, index=X.columns)
# importance.sort_values().plot(kind='barh', figsize=(8,6))
# plt.title("Feature Importance")
# plt.show()

# plt.figure(figsize=(10,8))
# plt.plot(y_pred,label='Predicted points')
# plt.show()













import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split



np.random.seed(42)


#sample data creation
n_samples = 200

#note down features

area = np.random.randint(500,3500,n_samples) #area of each home
bedrooms = np.random.randint(1,10,n_samples) #total bed rooms
age = np.random.randint(0,70,n_samples) #How much old house it would be 


#target specification
#price = more area+more bedrooms = high price ,more age = lower price

price = (area * 80)+(bedrooms* 50000)-(age*5000) + np.random.normal(0,20000,n_samples)

X = np.column_stack((area,bedrooms,age))
y = price
print(X.shape)

feature_names = ['Area (sqft)','BedRooms','Age (yrs)']

#model splitting and traning the model

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model training
GradinetModel = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.5,
    max_depth=3,
    random_state=42
)
GradinetModel.fit(X_train,y_train)


#prediction
y_prediction = GradinetModel.predict(X_test)


#model evaluation
print("r2_score of  test model ",r2_score(y_prediction,y_test))

#predictions
print(f"ALl predictions {y_prediction}")


#compare single tree and decision regression
single_tree = DecisionTreeRegressor(max_depth=3,random_state=42)
single_tree.fit(X_train,y_train)

tree_pred = single_tree.predict(X_test)


tree_mae = mean_absolute_error(y_test,tree_pred)
tree_r2 = r2_score(y_test,tree_pred)

#check model improve predicion

train_errors = []
test_errprs = []
tree_count = range(1,101)

for i,y_pred_staged in enumerate(GradinetModel.staged_predict(X_test),1):
    test_errprs.append(mean_absolute_error(y_test,y_pred_staged))

for i,y_pred_staged in enumerate(GradinetModel.staged_predict(X_train),1):
    test_errprs.append(mean_absolute_error(y_train,y_pred_staged))


for n in [1,5,10,30,50,100]:
    print(f" After the {n:>3} trees ----> test errors: {test_errprs[n-1]:.2f}")


#feature importance what exactly matters
importances_ = GradinetModel.feature_importances_
print(importances_)
