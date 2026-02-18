import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

#linear models 

#formulae = y=mx+b

# Random generation 
X = np.random.rand(200,3)
y = 4*X[:,0] - 2*X[:,1]+ 3*X[:,2]+np.random.randn(200)*0.2 

#train and test split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#model selection
model = LinearRegression()
model.fit(X_train,y_train)

#prediction and plottting
y_pred = model.predict(X_test)

test_data = model.predict(X_test)
train_data = model.predict(X_train)

print(f"mean square error test data {mean_squared_error(y_test,test_data)}")
print(f"mean square error train data {mean_squared_error(y_train,train_data)}")
print(f" r2 error test data {r2_score(y_test,test_data)}")
print(f"r2 error train data {r2_score(y_train,train_data)}")


print(f"All predictions are {y_pred}")

#plot the data
# plt.figure(figsize=(10,8))
# plt.scatter(y_test,y_pred,label='Orginal data',color='red')
# plt.plot(y_pred,label='predicted data',color='blue',linewidth=2)
# plt.xlabel("Orginal age ")
# plt.ylabel("salary hike")
# plt.legend()
# plt.show()

#metrics got for linear regression
# mean square error test data 0.041588477618473146
# mean square error train data 0.04151308533000987
#  r2 error test data 0.9805077942568056
# r2 error train data 0.9843833636589749

#no overfitting no underfitting found