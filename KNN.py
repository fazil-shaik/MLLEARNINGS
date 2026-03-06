import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error
import matplotlib.pyplot as plt

data = {
    'SqFt': [1500, 1800, 2400, 3000, 3500, 1200, 5000, 2200, 2800, 4000],
    'Age': [10, 15, 5, 2, 20, 30, 1, 12, 8, 15],
    'Price': [250, 280, 400, 550, 500, 180, 900, 350, 480, 650]
}

df = pd.DataFrame(data)

# 2. Split Features and Target
X = df[['SqFt', 'Age']]
y = df['Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_scaled,y)


y_prediction = knn.predict(X_scaled)

print(f"all predicitiona are {y_prediction}")
new_house = np.array([[2500, 12]])
new_house_scaled = scaler.transform(new_house) # Must scale the input too!

prediction = knn.predict(new_house_scaled)

distances, indices = knn.kneighbors(new_house_scaled)
neighbors = X.iloc[indices[0]]

print(f"Predicted Price for the house: ${prediction[0]:.2f}k")
print(f"r2_score is {r2_score(y,y_prediction)}")

plt.figure(figsize=(10, 6))
# Plot training data
scatter = plt.scatter(df['SqFt'], df['Age'], c=df['Price'], cmap='viridis', s=100, label='Training Data', alpha=0.6)
plt.colorbar(scatter, label='Price ($1000s)')

# Plot neighbors with a highlight
plt.scatter(neighbors['SqFt'], neighbors['Age'], edgecolors='red', facecolors='none', s=200, linewidths=2, label='3 Nearest Neighbors')

# Plot the new house
plt.scatter(new_house[0,0], new_house[0,1], color='red', marker='*', s=300, label=f'New House (Pred: ${prediction[0]:.0f}k)')

plt.xlabel('Square Footage')
plt.ylabel('Age (Years)')
plt.title('KNN Regression: Predicting House Price based on SqFt and Age')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('knn_regression_plot.png')