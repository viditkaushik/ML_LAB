import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

df = pd.read_csv("iris.csv")
X = df.drop("target", axis=1)
X = X.iloc[:, :2]

# CHANGE NUMBER OF CLUSTERS HERE
# Example: 2, 3, 4

k_value = 3

model = KMeans( n_clusters=k_value, random_state=42)

clusters = model.fit_predict(X)

print("K Value:", k_value)

print("\nCluster Centers:")
print(model.cluster_centers_)

plt.scatter( X.iloc[:,0], X.iloc[:,1], c=clusters)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("K-Means Clustering")

plt.show()