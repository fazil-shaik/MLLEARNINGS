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

#K-means clutsering 


from sklearn.cluster import KMeans
import numpy as np


data = np.array([[20, 80], [25, 85], [30, 90], [100, 20], [110, 25], [120, 30]])

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)
print(kmeans.labels_)

plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.legend()
plt.show()


#Hirearchial ck=lustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Sample data
data = [[1, 2], [2, 3], [8, 7], [8, 8], [25, 80]]

# Create linkage matrix for dendrogram
linkage_matrix = linkage(data, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Apply clustering
clustering = AgglomerativeClustering(n_clusters=3)
clusters = clustering.fit_predict(data)


from sklearn.cluster import DBSCAN
import numpy as np

# Sample data with noise
data = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80], [100, 100]])

# Apply DBSCAN
dbscan = DBSCAN(eps=3, min_samples=2)
clusters = dbscan.fit_predict(data)

plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.show()

print("Cluster assignments:", clusters)
print("Number of clusters:", len(set(clusters)) - (1 if -1 in clusters else 0))
print("Number of noise points:", list(clusters).count(-1))