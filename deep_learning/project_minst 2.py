import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Load data
mnist = load_digits()
X = mnist.data
y = mnist.target

# Perform K-means clustering
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    max_iter=1000,
    random_state=42,
    algorithm="elkan"
)

kmeans.fit(X)
labels = kmeans.labels_

# Print metrics (your existing code)
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

Global_mean = X.mean()
Global_std = X.std()
print(f"Global Mean: {Global_mean:.4f}")
print(f"Global Std Dev: {Global_std:.4f}")

# VISUALIZATION: 5x5 Grid with Cluster Info

# Select 25 random samples
np.random.seed(42)
n_samples = 25
random_indices = np.random.choice(len(X), n_samples, replace=False)

# Create figure with 5x5 subplots
fig, axes = plt.subplots(5, 5, figsize=(15, 15))

# Color map for clusters
cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# Plot each image with cluster information
for i, idx in enumerate(random_indices):
    row = i // 5
    col = i % 5
    ax = axes[row, col]
    
    # Reshape and display image
    image = X[idx].reshape(8, 8)
    cluster = labels[idx]
    actual_digit = y[idx]
    
    # Display with cluster color border
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Cluster: {cluster}\nActual: {actual_digit}', 
                 fontsize=10, color=cluster_colors[cluster % len(cluster_colors)])
    ax.axis('off')
    
    # Add colored border based on cluster
    for spine in ax.spines.values():
        spine.set_edgecolor(cluster_colors[cluster % len(cluster_colors)])
        spine.set_linewidth(3)

plt.suptitle('K-Means Clustering Results on Digits Dataset (k=3)', 
             fontsize=16, y=0.98)
plt.tight_layout()
plt.show()


# 1. Cluster distribution bar chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cluster size distribution
ax1 = axes[0, 0]
cluster_sizes = pd.Series(labels).value_counts().sort_index()
ax1.bar(cluster_sizes.index, cluster_sizes.values, 
        color=[cluster_colors[i % len(cluster_colors)] for i in cluster_sizes.index])
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Number of Samples')
ax1.set_title('Cluster Size Distribution')
ax1.set_xticks(cluster_sizes.index)
for i, v in enumerate(cluster_sizes.values):
    ax1.text(i, v + 5, str(v), ha='center', va='bottom')

# Plot 2: Actual digit distribution within clusters
ax2 = axes[0, 1]
cluster_digit_counts = pd.crosstab(labels, y)
cluster_digit_counts.plot(kind='bar', stacked=True, ax=ax2, 
                          colormap='tab10', legend=True)
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Count')
ax2.set_title('Digit Distribution by Cluster')
ax2.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 3: Cluster centroids visualization
ax3 = axes[1, 0]
centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    ax3.imshow(centroid.reshape(8, 8), cmap='gray', 
               extent=[i, i+1, 0, 1])
    ax3.text(i+0.5, 1.1, f'Cluster {i}', ha='center', va='bottom')
ax3.set_xlim(0, len(centroids))
ax3.set_ylim(0, 1.2)
ax3.set_title('Cluster Centroids')
ax3.axis('off')

# Plot 4: Scatter plot using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
scatter = ax4 = axes[1, 1]
scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=labels, cmap='viridis', alpha=0.6, s=10)
ax4.set_xlabel('First Principal Component')
ax4.set_ylabel('Second Principal Component')
ax4.set_title('Data Points Colored by Cluster (PCA)')
plt.colorbar(scatter, ax=ax4, label='Cluster')

plt.tight_layout()
plt.show()


# Create confusion matrix between clusters and actual digits
fig, ax = plt.subplots(figsize=(10, 8))
confusion_matrix = pd.crosstab(labels, y)
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Actual Digit')
ax.set_ylabel('Cluster')
ax.set_title('Confusion Matrix: Clusters vs Actual Digits')
plt.tight_layout()
plt.show()


from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

fig, ax = plt.subplots(figsize=(10, 7))

# Get silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, labels)

y_lower = 10
for i in range(kmeans.n_clusters):
    # Aggregate silhouette scores for samples in cluster i
    ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()
    
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.nipy_spectral(float(i) / kmeans.n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    # Label the silhouette plots with their cluster numbers at the middle
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax.set_xlabel("Silhouette Coefficient Values")
ax.set_ylabel("Cluster Label")
ax.set_title("Silhouette Plot for K-Means Clustering")
ax.axvline(x=silhouette, color="red", linestyle="--", 
           label=f"Average: {silhouette:.3f}")
ax.set_yticks([])  # Clear the yaxis labels / ticks
ax.legend(loc="best")

plt.tight_layout()
plt.show()