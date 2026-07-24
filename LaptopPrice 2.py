import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    'ram_gb': [4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16],

    'storage_gb': [256, 512, 128, 512, 256, 512, 256, 1024, 256, 512,
                   128, 512, 256, 1024, 256, 512, 128, 512, 256, 1024,
                   256, 512, 128, 512, 256, 512, 256, 1024, 256, 512],

    'processor_ghz': [2.1, 2.8, 1.8, 3.2, 2.4, 3.0, 2.0, 3.5, 2.6, 3.0,
                      1.6, 2.8, 2.2, 3.4, 2.5, 2.9, 1.9, 3.1, 2.3, 3.6,
                      2.0, 2.7, 1.7, 3.3, 2.4, 3.0, 2.1, 3.5, 2.6, 3.2],

    'price_inr': [28000, 45000, 22000, 72000, 38000, 52000, 26000, 95000, 42000, 68000,
                  20000, 48000, 29000, 88000, 40000, 50000, 23000, 70000, 36000, 98000,
                  25000, 46000, 21000, 75000, 39000, 53000, 27000, 92000, 43000, 73000]
}

df = pd.DataFrame(data)

# Your code starts here...

# Features and target variable
X = df[['ram_gb', 'storage_gb', 'processor_ghz']]
y = df['price_inr']
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
print(f"-----> Coefficient for RAM (GB): {model.coef_[0]:.2f} INR per GB")
print(f"-----> Coefficient for Storage (GB): {model.coef_[1]:.2f} INR per GB")
print(f"-----> Coefficient for Processor Speed (GHz): {model.coef_[2]:.2f} INR per GHz")
# Predict price for a new laptop configuration
new_laptop = pd.DataFrame({'ram_gb': [8], 'storage_gb': [512], 'processor_ghz': [2.5]})
predicted_price = model.predict(new_laptop)
print(f"\n--------- New Laptop Price Prediction -----------")
print(f"RAM: 8 GB")
print(f"Storage: 512 GB")
print(f"Processor Speed: 2.5 GHz")
print(f"Predicted Price: {predicted_price[0]:.1f} INR")
# Visualize the relationship between features and price
fig, axis = plt.subplots(1, 3, figsize=(14, 5))
features = [df['ram_gb'], df['storage_gb'], df['processor_ghz']]
names = ['RAM (GB)', 'Storage (GB)', 'Processor Speed (GHz)']
colors = ["#FFA500", "#800080", "#008000"]
for i, (feature, name, color) in enumerate(zip(features, names, colors)):
    axis[i].scatter(feature, df['price_inr'], color=color, s=100, alpha=0.7)
    axis[i].set_ylabel("Price (INR)", fontsize=11)
    axis[i].set_xlabel(name, fontsize=11)
    axis[i].set_title(f"{name} vs Price", fontsize=14)
plt.tight_layout()
plt.show()