import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
# READ CSV
df = pd.read_csv("iris.csv")

# REMOVE TARGET COLUMN
X = df.drop("target", axis=1)

# TAKE FIRST 2 FEATURES
X = X.iloc[:, :2]

model1 = DBSCAN(eps=0.3,min_samples=5)
labels1 = model1.fit_predict(X)
plt.figure(figsize=(5,5))
plt.scatter(X.iloc[:,0],X.iloc[:,1],c=labels1)

plt.title("DBSCAN eps=0.3 min_samples=5")
plt.show()

model2 = DBSCAN(eps=0.5,min_samples=3)
labels2 = model2.fit_predict(X)

plt.figure(figsize=(5,5))

plt.scatter(X.iloc[:,0],X.iloc[:,1],c=labels2)
plt.title("DBSCAN eps=0.5 min_samples=3")
plt.show()