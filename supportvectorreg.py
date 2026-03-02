from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

np.random.seed(42)

#generate data
X = np.sort(5*np.random.rand(100,1),axis=0)
y = np.sin(X).ravel()

#add noise 
y[::5] += 2*(0.5 - np.random.rand(20))

#model traing and tetsing

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection

SuportModel = SVR(kernel='rbf',C=100,gamma=0.1,epsilon=0.1)
SuportModel.fit(X_train,y_train)

#prediction
y_pred = SuportModel.predict(X_test)

#evaluation
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

X_test = np.linspace(0, 5, 100).reshape(-1,1)
y_pred = SuportModel.predict(X_test)

#plotting
plt.scatter(X, y, color='darkorange', label='Data')
plt.plot(X_test, y_pred, color='navy', label='SVR model')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Support Vector Regression')
plt.legend()
plt.show()