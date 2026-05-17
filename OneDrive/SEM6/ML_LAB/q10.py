import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# READ CSV
df = pd.read_csv("iris.csv")
X = df.drop("target", axis=1)

y = df["target"]

# CHANGE NUMBER OF COMPONENTS HERE
# Example: 2, 3

components = 2

pca = PCA(n_components=components)

X_pca = pca.fit_transform(X)

print("Number of Components:", components)
print("\nExplained Variance Ratio:")

print(pca.explained_variance_ratio_)

plt.scatter( X_pca[:,0],X_pca[:,1],c=y)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")

plt.show()