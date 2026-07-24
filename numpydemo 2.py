import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


datadigits = load_digits()

X = datadigits.data
y = datadigits.target

print("Shape of the data matrix X:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

KmeansModel = KMeans(n_clusters=10, random_state=42)
KmeansModel.fit(X_train)


y_predict = KmeansModel.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))


plt.acorr(y_test - y_predict, maxlags=10)
plt.title("Autocorrelation of Residuals")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()

plt.scatter(y_test, y_predict)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")  
plt.show()

plt.plot(KmeansModel.inertia_)
plt.xticks(range(1, len(KmeansModel.inertia_) + 1))
plt.grid()
plt.title("KMeans Inertia")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()



# df = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print("Original Array:")
# print(df)

# print("\nTranspose of the Array:")
# print(df.T)

# print("\nMatrix multiplication toolkit:")
# print(np.dot(df, df.T))

# print("reshaped ones are : \n")
# reshaped_array = np.reshape(df, (9, 1))
# print(reshaped_array)

# flatten_array = df.flatten()
# print("\nFlattened Array:")
# print(flatten_array)

# print("ravel array is : \n")
# ravel_array = df.ravel()
# print(ravel_array)

