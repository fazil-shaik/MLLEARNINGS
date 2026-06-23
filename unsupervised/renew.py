import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Generate Swiss Roll Dataset
X, t = make_swiss_roll(
    n_samples=1000,
    noise=0.05,
    random_state=42
)

# Create KMeans Model
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    max_iter=1000,
    random_state=42,
    algorithm="elkan"
)

# Train Model
kmeans.fit(X)

# Cluster Labels
labels = kmeans.labels_

# Evaluation Metrics
silhouette = silhouette_score(X, labels)
davies_bouldin = davies_bouldin_score(X, labels)
calinski_harabasz = calinski_harabasz_score(X, labels)

print("=" * 50)
print("K-MEANS CLUSTERING RESULTS")
print("=" * 50)

print(f"Inertia (SSE): {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")

print("\nCluster Centers:")
print(kmeans.cluster_centers_)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X[:, 0],
    X[:, 1],
    X[:, 2],
    c=labels,
    cmap="viridis",
    s=20
)

# Plot cluster centers
ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    kmeans.cluster_centers_[:, 2],
    marker="X",
    s=300
)

ax.set_title("K-Means Clustering on Swiss Roll Dataset")

plt.show()