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



#DBScan clsustering 
from sklearn.cluster import DBSCAN
import numpy as np

data = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80], [100, 100]])

# Apply DBSCAN
dbscan = DBSCAN(eps=3, min_samples=2)
clusters = dbscan.fit_predict(data)

print("Cluster assignments:", clusters)
print("Number of clusters:", len(set(clusters)) - (1 if -1 in clusters else 0))
print("Number of noise points:", list(clusters).count(-1))


plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.show()




#Association rules Appriori Algorithm
# from mlxtend.frequent_patterns import apriori, association_rules
# import pandas as pd

# # Sample transaction data
# transactions = [
#     ['bread', 'butter', 'jam'],
#     ['bread', 'butter'],
#     ['bread', 'jam'],
#     ['butter', 'jam'],
#     ['bread', 'butter', 'jam', 'milk'],
#     ['bread', 'milk'],
#     ['butter', 'milk']
# ]

# Convert to binary matrix
# from mlxtend.preprocessing import TransactionEncoder
# te = TransactionEncoder()
# te_ary = te.fit(transactions).transform(transactions)
# df = pd.DataFrame(te_ary, columns=te.columns_)

# # Find frequent itemsets
# frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# # Generate association rules
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
# print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# fig, ax = plt.subplots(0,3,figsize=(15,5))
# ax[0].scatter(rules['support'], rules['confidence'], alpha=0.5)
# ax[0].set_xlabel('Support')
# ax[0].set_ylabel('Confidence')
# ax[0].set_title('Support vs Confidence')
# ax[1].scatter(rules['support'], rules['lift'], alpha=0.5)
# ax[1].set_xlabel('Support')
# ax[1].set_ylabel('Lift')
# ax[1].set_title('Support vs Lift')
# ax[2].scatter(rules['confidence'], rules['lift'], alpha=0.5)
# ax[2].set_xlabel('Confidence')
# ax[2].set_ylabel('Lift')
# ax[2].set_title('Confidence vs Lift')
# plt.tight_layout()
# plt.show()



# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor
# import numpy as np

# # Sample data with outliers
# np.random.seed(42)
# normal_data = np.random.normal(0, 1, (100, 2))
# outliers = np.random.uniform(-4, 4, (5, 2))
# data = np.vstack([normal_data, outliers])

# # Isolation Forest
# iso_forest = IsolationForest(contamination=0.1, random_state=42)
# outlier_labels = iso_forest.fit_predict(data)

# print("Outlier detection results:")
# print("Normal points:", np.sum(outlier_labels == 1))
# print("Anomalies:", np.sum(outlier_labels == -1))

# # Local Outlier Factor
# lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# outlier_labels_lof = lof.fit_predict(data)

#GMM
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data1 = np.random.normal([2, 2], [0.5, 0.5], (50, 2))
data2 = np.random.normal([6, 6], [1, 1], (50, 2))
data = np.vstack([data1, data2])

# Apply GMM
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data)

# Get cluster assignments and probabilities
cluster_labels = gmm.predict(data)
probabilities = gmm.predict_proba(data)

print("Cluster means:", gmm.means_)
print("Cluster covariances:", gmm.covariances_)
print("Sample probabilities:", probabilities[:5])


#customer segmentation via K-means clustering

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
 
print("\n" + "="*80)
print("PROBLEM 1: Customer Segmentation via K-means")
print("="*80)
 
def segment_customers(df):
    """
    Segment customers based on RFM (Recency, Frequency, Monetary).
    
    Input:
      df: DataFrame with columns [customer_id, purchase_amount, purchase_date]
    
    Output:
      DataFrame with cluster assignments and business interpretation
    """
    # Calculate RFM metrics
    now = pd.Timestamp.now()
    
    rfm = df.groupby('customer_id').agg({
        'purchase_amount': ['sum', 'count', 'mean'],  # M (Monetary), F (Frequency), AOV
        'purchase_date': lambda x: (now - x.max()).days  # R (Recency)
    }).reset_index()
    
    rfm.columns = ['customer_id', 'total_spent', 'purchase_count', 'avg_order_value', 'days_since_purchase']
    
    # Select features and scale
    features = ['days_since_purchase', 'purchase_count', 'total_spent']
    X = rfm[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal K using silhouette score
    best_k = 2
    best_score = -1
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"K={k}: Silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k
    
    # Fit final model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm['segment'] = kmeans.fit_predict(X_scaled)
    
    # Label segments by monetary value
    segment_spending = rfm.groupby('segment')['total_spent'].mean().sort_values(ascending=False)
    segment_map = {seg: f"Tier_{i+1}" for i, seg in enumerate(segment_spending.index)}
    rfm['segment_name'] = rfm['segment'].map(segment_map)
    
    print(f"\n✓ Optimal K: {best_k} (Silhouette: {best_score:.3f})")
    print(f"\nSegment breakdown:")
    print(rfm.groupby('segment_name')[['total_spent', 'purchase_count', 'days_since_purchase']].agg({
        'total_spent': ['count', 'mean'],
        'purchase_count': 'mean',
        'days_since_purchase': 'mean'
    }).round(2))
    
    return rfm[['customer_id', 'segment_name', 'total_spent', 'purchase_count', 'days_since_purchase']]
 
# Example usage
np.random.seed(42)
example_customers = pd.DataFrame({
    'customer_id': range(1, 501),
    'purchase_amount': np.random.lognormal(5, 1.5, 500),  # Realistic spend distribution
    'purchase_date': pd.date_range('2023-01-01', periods=500, freq='2h')
})
# Duplicate some customers
example_customers = pd.concat([example_customers, example_customers.sample(200)])
 
segmented = segment_customers(example_customers.reset_index(drop=True))
print("\nSample segmented customers:")
print(segmented.head(10))
 