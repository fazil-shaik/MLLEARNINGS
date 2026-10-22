import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "customer": [
        "A", "B", "C", "D", "E", "F",
        "G", "H", "I", "J", "K", "L"
    ],
    "income": [
        3, 4, 5, 4,
        8, 9, 10, 8,
        12, 13, 14, 12
    ],
    "spending": [
        15, 20, 25, 18,
        45, 50, 55, 48,
        80, 85, 90, 82
    ]
}

df = pd.DataFrame(data)

print(df)


X = df[["income", "spending"]]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42
)

kmeans.fit(X_scaled)

df["cluster"] = kmeans.labels_

print(df)


centroids_scaled = kmeans.cluster_centers_

centroids_original = scaler.inverse_transform(
    centroids_scaled
)

centroid_df = pd.DataFrame(
    centroids_original,
    columns=["income", "spending"]
)

print(centroid_df)



plt.figure(figsize=(9, 6))

plt.scatter(
    X["income"],
    X["spending"],
    c=df["cluster"],
    s=100
)

plt.scatter(
    centroid_df["income"],
    centroid_df["spending"],
    marker="X",
    s=250,
    edgecolors="black"
)

plt.xlabel("Annual Income (₹ lakh)")
plt.ylabel("Annual Spending (₹ thousand)")
plt.title("Customer Segmentation Using K-Means")

for i, customer in enumerate(df["customer"]):
    plt.annotate(
        customer,
        (X.iloc[i]["income"], X.iloc[i]["spending"])
    )

plt.show()