import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("datasets/iris1.csv")

X = StandardScaler().fit_transform(df.iloc[:, :4])

# AGNES
agnes = AgglomerativeClustering(n_clusters=3, linkage="ward")
agnes_labels = agnes.fit_predict(X)

# DIANA (using Bisecting K-Means)
diana = BisectingKMeans(n_clusters=3, random_state=42)
diana_labels = diana.fit_predict(X)

# Dendrogram
linkage_matrix = linkage(X, method="ward")
dendrogram(linkage_matrix, no_labels=True)
plt.title("Dendrogram")
plt.show()

# AGNES Plot
plt.scatter(X[:,0], X[:,1], c=agnes_labels)
plt.title("AGNES Clustering")
plt.show()

# DIANA Plot
plt.scatter(X[:,0], X[:,1], c=diana_labels)
plt.title("DIANA Clustering (Bisecting K-Means)")
plt.show()

print("AGNES Cluster Labels:")
print(agnes_labels)

print("\nDIANA Cluster Labels:")
print(diana_labels)
