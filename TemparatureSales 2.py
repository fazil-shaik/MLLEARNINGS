import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#data collection

temperature = np.array([25,28,32,35,38,40,42])
sales = np.array([10,14,18,20,24,26,28])

#numpy data
x = temperature.reshape(-1,1)
y = sales

#create model

model = LinearRegression()

model.fit(x,y)

#predit_sales

new_tempeature = 30
preditcted_sales = model.predict(np.array([[new_tempeature]]))

print(f"The predicted sales for temp {temperature}  is: {preditcted_sales[0]:.0f}")

#see data
plt.figure(figsize=(10,6))
plt.savefig('temperatureVsSales.png',dpi=300)
plt.scatter(temperature,sales,color='green',s=100,label='Actaual data')
plt.scatter(new_tempeature,preditcted_sales,color='red',s=200,label='Predicted')
plt.plot(temperature,model.predict(x),color='blue',linewidth=2,label='predicted value')
plt.title('Temp vs sales Prediction',fontsize=16)
plt.xlabel('Temp Found',fontsize=14)
plt.ylabel('sales Obtained',fontsize=14)
plt.show()


