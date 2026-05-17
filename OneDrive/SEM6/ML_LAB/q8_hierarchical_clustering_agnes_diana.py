import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

df = pd.read_csv("datasets/iris1.csv")
X = StandardScaler().fit_transform(df[df.columns[:4]])

# ---- AGNES (Agglomerative - bottom-up) ----
agnes = AgglomerativeClustering(n_clusters=3, linkage="ward")
agnes_labels = agnes.fit_predict(X)

# ---- DIANA (Divisive - top-down, custom) ----
def diana(X, n_clusters):
    clusters = [list(range(len(X)))]
    dist_matrix = squareform(pdist(X))
    
    while len(clusters) < n_clusters:
        # Split the largest cluster
        largest = max(clusters, key=len)
        if len(largest) < 2:
            break
        clusters.remove(largest)
        
        # Find most dissimilar point (average distance to others in cluster)
        avg_dists = [np.mean([dist_matrix[i][j] for j in largest if j != i]) for i in largest]
        splinter_idx = largest[np.argmax(avg_dists)]
        
        group1, group2 = [splinter_idx], [p for p in largest if p != splinter_idx]
        # Reassign based on closer group
        changed = True
        while changed:
            changed = False
            for p in group2[:]:
                d1 = np.mean([dist_matrix[p][q] for q in group1])
                d2 = np.mean([dist_matrix[p][q] for q in group2 if q != p]) if len(group2) > 1 else np.inf
                if d1 < d2:
                    group1.append(p); group2.remove(p); changed = True
        clusters.extend([group1, group2])
    
    labels = np.zeros(len(X), dtype=int)
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = i
    return labels

diana_labels = diana(X, n_clusters=3)

# ---- Plots ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Dendrogram (for AGNES)
Z = linkage(X, method="ward")
axes[0].set_title("Dendrogram (AGNES)")
dendrogram(Z, ax=axes[0], truncate_mode="lastp", p=15)
axes[0].set_xlabel("Samples"); axes[0].set_ylabel("Distance")

# AGNES scatter
for c in range(3):
    mask = agnes_labels == c
    axes[1].scatter(X[mask, 0], X[mask, 1], label=f"C{c+1}", s=30)
axes[1].set_title("AGNES Clustering"); axes[1].legend()
axes[1].set_xlabel("Sepal Length"); axes[1].set_ylabel("Petal Length")

# DIANA scatter
for c in range(3):
    mask = diana_labels == c
    axes[2].scatter(X[mask, 0], X[mask, 1], label=f"C{c+1}", s=30)
axes[2].set_title("DIANA Clustering"); axes[2].legend()
axes[2].set_xlabel("Sepal Length"); axes[2].set_ylabel("Petal Length")

plt.tight_layout(); plt.savefig("q8_hierarchical.png", dpi=100); plt.show()
print("AGNES label counts:", {i: np.sum(agnes_labels==i) for i in range(3)})
print("DIANA label counts:", {i: np.sum(diana_labels==i) for i in range(3)})
