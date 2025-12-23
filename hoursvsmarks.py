import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#data
hours = np.array([2,3,4,5,6,7,8])
marks = np.array([35,55,65,78,88,90,95])

#data in numpy
x = hours.reshape(-1,1)
y = marks

#create a model

model = LinearRegression()
model.fit(x,y)

#predit marks
new_hours = 5.4
predicted_marks = model.predict(np.array([[new_hours]]))
print(f"The predicted marks for studying {new_hours} hours is: {predicted_marks[0]:.0f}")

#see data
plt.figure(figsize=(10,6))
plt.scatter(hours,marks,color='green',s=100,label='Actaual data')
plt.scatter(new_hours,predicted_marks,color='red',s=200,label='Predicted')
plt.plot(hours,model.predict(x),color='blue',linewidth=2,label='predicted value')
plt.title('Hours vs Marks Prediction',fontsize=16)
plt.xlabel('Hours Studied',fontsize=14)
plt.ylabel('Marks Obtained',fontsize=14)
plt.show()



