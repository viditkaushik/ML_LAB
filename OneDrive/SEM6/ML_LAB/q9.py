import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

# READ CSV
df = pd.read_csv("iris.csv")
X = df.drop("target", axis=1)

# TAKE FIRST 2 FEATURES
X = X.iloc[:, :2]

# CHANGE eps VALUE HERE
# Example: 0.3, 0.5, 0.7

eps_value = 0.5

# CHANGE min_samples VALUE HERE
# Example: 3, 5, 10

min_samples_value = 5

model = DBSCAN(eps=eps_value,min_samples=min_samples_value)

labels = model.fit_predict(X)

print("eps =", eps_value)
print("min_samples =", min_samples_value)

print("\nCluster Labels:")
print(labels)

plt.scatter( X.iloc[:,0],X.iloc[:,1],c=labels)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")
plt.show()