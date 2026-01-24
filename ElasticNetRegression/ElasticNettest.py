from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt


np.random.seed(42)

# Generate synthetic data
X = np.random.rand(100,2)
y = np.random.rand(100,2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ElasticNet_model = ElasticNet(alpha=0.1,l1_ratio=0.5,random_state=42,max_iter=1000)

ElasticNet_model.fit(X_train,y_train)

y_pred = ElasticNet_model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2value = r2_score(y_test,y_pred)

print(f"Linear ElasticNet Model Performace: ")

print(f"mearn squre error: {mse}")
print(f"R2 score: {r2value}")

plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,color='blue',label='predcited vs actual',alpha=0.5)
plt.plot([0,1],[0,1],color='red',linestyle='--',label='ideal fit')
plt.plot(X_test,y_test,color='green',label='actual values',alpha=0.3)
plt.xlabel('Actual values')
plt.ylabel('predicted values')
plt.title('ElasticNet regression')
plt.legend()
plt.show()