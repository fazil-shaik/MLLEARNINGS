import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#data 
yoe = np.array([1,2,3,5,7,10])
salary = np.array([4.5,6.0,7.5,10.0,12.5,15.0])

#reshape the data
X = yoe.reshape(-1,1)
Y = salary

#model training 
model = LinearRegression()
model.fit(X,Y)

#check prediecations
newYoe = 8
predicted_salary = model.predict(np.array([[newYoe]]))
print(f"Predicted salary for {newYoe} years of experience: {predicted_salary[0]:.2f}K")