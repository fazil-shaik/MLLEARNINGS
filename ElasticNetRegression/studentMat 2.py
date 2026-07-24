import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet,LinearRegression,ridge_regression,Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

URL = "https://storage.googleapis.com/kagglesdsdata/datasets/4312217/7413268/student-mat.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260124%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260124T033739Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9608d3037b8d4cf4f56660ac605a79e020a322d8168a0bcaf7a69b0f410e8a9777093b6f645840a5fed82528670615f8cb7eed5c4c38b92d5129c0a19f31995c60c0fdaf467f5508d6e6ac51314b3bf7458828e3f201334908931a8e9356808acd087b86d9b029b5d11de73be772c87a52cb617310b4f5bfead94dc640647a160664093c92ea9501e2173d211d37ecebf1b4251389a3c7d758f2cdcddee1fcec011865f641f2f1db6e42dd43e847c72bb15008b97047377a0f8e0b92be46129451c8affb6f9882a3d05187615c6ea8dab2ec355fbb6432288b2e6741b7fe6fa641175b46e9fa989ad1a527a08f9a33dafed3b4ee20907b98f1076d9f366b21ef"

data_value = pd.read_csv(URL)

getFirst_five_coloumns = data_value.iloc[:, :5]

print(getFirst_five_coloumns.head())

X = np.array(data_value.drop(['G3'], axis=1).select_dtypes(include=[np.number]))
y = np.array(data_value['G3'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

model = ElasticNet(alpha=1.0,l1_ratio=0.5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Predictions: {y_pred}")


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")
plt.plot([0, 20], [0, 20], color='red',marker='o')
plt.plot(X,y,color='purple',label='Orginal data')
plt.plot(y_pred,color='green',label='Fitted line')
plt.show()  
print("ElasticNet Regression model training and evaluation completed.") 
