from sklearn.datasets import load_iris # Dataset
from sklearn.decomposition import PCA # Algorithm
import matplotlib.pyplot as plt # Visualization
from sklearn.cluster import KMeans
# Load the data 
iris_data = load_iris(as_frame=True)

# Preview
print(iris_data.data.head())

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sepal_length = iris_data.data["sepal length (cm)"]
sepal_width = iris_data.data["sepal width (cm)"]
petal_length = iris_data.data["petal length (cm)"]
petal_width = iris_data.data["petal width (cm)"]

ax.scatter(sepal_length, sepal_width, petal_length, c=petal_width)
plt.show()

X = iris_data.data.values

pca = PCA(n_components=2)

# Train the model 
pca.fit(iris_data.data)
iris_data_reduced = pca.fit_transform(iris_data.data)

# Plot data
plt.scatter(
    iris_data_reduced[:,0],
    iris_data_reduced[:,1],
    c=iris_data.target
)
plt.show()


Kmeans = KMeans(n_clusters=3, random_state=0)
Kmeans.fit(iris_data.data)
print(Kmeans.labels_)
labels = Kmeans.labels_

plt.scatter(X[:, 0], X[:, 2], c=labels, cmap='viridis', marker='o')
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 2], 
            c='red', marker='x', s=200, label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('K-Means Clustering on Iris Dataset')
plt.legend()
plt.show()