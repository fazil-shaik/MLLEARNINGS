import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(data)

# Your code starts here...

# Features and target variable
X = df[['distance_km', 'prep_time_min']]
y = df['delivery_time_min']
print("Features (X):")
print(X)
print(f"\nShape of X: {X.shape}")
print("Target variable (y):")
print(y)
print(f"\nShape of y: {y.shape}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
print(f"\n----- Model Training Complete -----")
print(f"Coefficients: {model.coef_.round(2)}")
print(f"-----> Coefficient for distance_km: {model.coef_[0]:.2f} minutes per km")
print(f"-----> Coefficient for prep_time_min: {model.coef_[1]:.2f} minutes per minute")
# Predict delivery time for a new order
new_order = pd.DataFrame({'distance_km': [5.0], 'prep_time_min': [15]})
predicted_delivery_time = model.predict(new_order)
print(f"\n--------- New Order Prediction -----------")
print(f"Distance: 5.0 km")
print(f"Preparation Time: 15 minutes")
print(f"Predicted Delivery Time: {predicted_delivery_time[0]:.1f} minutes")
# Visualize the relationship between features and delivery time
fig, axis = plt.subplots(1, 2, figsize=(14, 5))
features = [df['distance_km'], df['prep_time_min']]
names = ['Distance (km)', 'Preparation Time (min)']
colors = ["#FF5733", "#33C1FF"]
for i, (feature, name, color) in enumerate(zip(features, names, colors)):
    axis[i].scatter(feature, df['delivery_time_min'], color=color, s=100, alpha=0.7)
    axis[i].set_ylabel("Delivery Time (min)", fontsize=11)
    axis[i].set_xlabel(name, fontsize=11)
    axis[i].set_title(f"Delivery Time vs {name}", fontsize=14)
plt.show()  