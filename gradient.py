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





from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



np.random.seed(42)


#sample data creation
n_samples = 500

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


param_grid = {
    'n_estimators': [300, 500, 800],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [2, 3, 4],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best R2:", grid.best_score_)

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
# print(f"ALl predictions {y_prediction}")


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


for name, imp in sorted(zip(feature_names,importances_),key=lambda x:-x[1]):
    bar = '|'*int(imp*50)
    print(f"{name:<15}:{imp:.3f} {bar}")



#custom predicitons
custome_house = np.array([[2000,3,10]])
pred_cust_house = GradinetModel.predict(custome_house)


fig,axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle('Gradient regression booster')


#axes1-plotting actual vs predicted
ax1 = axes[0,0]
ax1.scatter(y_test,y_prediction,color='blue',edgecolor='white',alpha=0.6)
min_val = min(min(y_test),min(y_prediction))
max_val = max(max(y_test),max(y_prediction))
ax1.plot([min_val,max_val],[min_val,max_val],'r--')
ax1.set_title('Actual vs predicted Price')
ax1.set_xlabel('Actual price')
ax1.set_ylabel('Predicted price')
ax1.legend()


# AX2 - Feature Importance
ax2 = axes[0,1]

sorted_idx = np.argsort(importances_)
ax2.barh(np.array(feature_names)[sorted_idx],
         importances_[sorted_idx],
         color='green')

ax2.set_title("Feature Importance")
ax2.set_xlabel("Importance Score")


# AX3 - Learning Curve (Train vs Test MAE)
ax3 = axes[1,0]

train_errors = []
test_errors = []

for y_pred_train, y_pred_test in zip(
        GradinetModel.staged_predict(X_train),
        GradinetModel.staged_predict(X_test)):

    train_errors.append(mean_absolute_error(y_train, y_pred_train))
    test_errors.append(mean_absolute_error(y_test, y_pred_test))

ax3.plot(train_errors, label='Train MAE', color='blue')
ax3.plot(test_errors, label='Test MAE', color='red')

ax3.set_title("Boosting Learning Curve")
ax3.set_xlabel("Number of Trees")
ax3.set_ylabel("MAE")
ax3.legend()

# AX4 - Model Comparison
ax4 = axes[1,1]

models = ['Decision Tree', 'Gradient Boosting']
r2_scores = [tree_r2, r2_score(y_test, y_prediction)]

ax4.bar(models, r2_scores, color=['orange','purple'])
ax4.set_title("Model R² Comparison")
ax4.set_ylabel("R² Score")
ax4.set_ylim(0,1)

for i, v in enumerate(r2_scores):
    ax4.text(i, v + 0.02, f"{v:.2f}", ha='center')


plt.show()