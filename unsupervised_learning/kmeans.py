
"""
==========================================================================
                        K-MEANS CLUSTERING
==========================================================================

Author : Alfred Joseph T

Theory
------
K-Means is an unsupervised learning algorithm that groups similar data
points into K clusters. It iteratively assigns points to the nearest
centroid and updates centroids until convergence.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================================
# Load Dataset
# ==========================================================

DATASET_PATH = "../../datasets/customer_segmentation.csv"

data = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("Customer Segmentation Dataset")
print("=" * 60)
print(data.head())
print("\nShape :", data.shape)
print("\nColumns :", list(data.columns))

# ==========================================================
# Select Features
# ==========================================================

X = data[["Annual_Income", "Spending_Score"]]

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================================
# Elbow Method
# ==========================================================

inertia = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================================
# Train K-Means
# ==========================================================

k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

print("\nCluster Counts")
print(data["Cluster"].value_counts().sort_index())

print("\nCluster Centers (Scaled)")
print(kmeans.cluster_centers_)

print("\nFinal Inertia :", kmeans.inertia_)

# ==========================================================
# Scatter Plot
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=clusters,
    cmap="viridis",
    s=40
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    c="red",
    marker="X",
    s=220,
    label="Centroids"
)

plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.title("K-Means Clustering")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================================
# Predict New Customer
# ==========================================================

new_customer = [[65, 80]]

new_scaled = scaler.transform(new_customer)

cluster = kmeans.predict(new_scaled)

print("\nNew Customer")
print("Annual Income :", new_customer[0][0])
print("Spending Score:", new_customer[0][1])
print("Assigned Cluster:", cluster[0])

# ==========================================================
# Cluster Summary
# ==========================================================

summary = data.groupby("Cluster")[[
    "Annual_Income",
    "Spending_Score"
]].mean()

print("\nCluster Summary")
print(summary)

# ==========================================================
# Conclusion
# ==========================================================

print("""
Advantages
----------
* Simple and fast
* Easy to implement
* Works well for spherical clusters

Disadvantages
-------------
* Must choose K beforehand
* Sensitive to outliers
* Sensitive to centroid initialization

Applications
------------
* Customer Segmentation
* Market Analysis
* Student Grouping
* Image Compression
* Recommendation Systems
""")

print("="*60)
print("Program Finished Successfully")
print("="*60)
