from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

SEED = 42

X = np.random.rand(100,10)
y = np.random.randint(0,9,100)

#model selection
model = LogisticRegression(random_state=SEED)
model.fit(X,y)

#predictin and eval
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.2f}")
print(f"Predicted labels: {y_pred}")


new_data = np.random.rand(5,10)
new_predictions = model.predict(new_data)
print(f"Predictions for new data: {new_predictions}")

#plotting
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k', s=100)
# plt.plot(X[:,0], X[:,1], 'o', color='red', markersize=10, label='Data Points')
# plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.title('Binary Classification Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class Label')
plt.show()

#new data plotting
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k', s=100, label='Training Data')
plt.scatter(new_data[:,0], new_data[:,1], c=new_predictions, cmap='coolwarm', edgecolor='k', s=100, marker='X', label='New Data')
plt.title('Binary Classification with New Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class Label')
plt.legend()
plt.show()