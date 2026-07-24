import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'ctr': [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
            4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
            7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'total_views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}

df = pd.DataFrame(data)

# Your code starts here...

X = np.array(df['ctr']).reshape(-1,1)
Y = np.array(df['total_views'])

#model training
model = LinearRegression()
model.fit(X,Y)

#predict total views
new_ctr = 1.8
predicted_views = model.predict(np.array([[new_ctr]]))

print(f"The predicted total views for a CTR of {new_ctr}% is: {predicted_views[0]:.0f}")        

#visualization
plt.figure(figsize=(10,6))
plt.scatter(df['ctr'], df['total_views'], color='green', s=100, label='Actual data')
plt.scatter(new_ctr, predicted_views, color='red', s=200, label='Predicted')
plt.plot(df['ctr'], model.predict(X), color='blue', linewidth=2, label='Predicted value')
plt.title('CTR vs Total Views Prediction', fontsize=16)
plt.xlabel('Click-Through Rate (CTR %)', fontsize=14)
plt.ylabel('Total Views', fontsize=14)
plt.legend()
plt.show()