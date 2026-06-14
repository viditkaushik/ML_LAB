import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- Load data from CSV ---
# Replace with your actual CSV path
df = pd.read_csv("datasets/iris.csv")

# Select features (here: first two columns for visualization)
X = df.iloc[:, :2].values  

# --- Elbow Method ---
wcss = []
K_range = range(1, 11)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=321)
    model.fit(X)
    wcss.append(model.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, wcss, "bo-")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal K")
plt.show()

# --- Choose optimal K (say 3 from elbow curve) ---
optimal_k = 3
model_opt = KMeans(n_clusters=optimal_k, random_state=321)
clusters = model_opt.fit_predict(X)

# --- Plot clusters ---
plt.figure(figsize=(6,5))
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.title(f"KMeans Clustering (K={optimal_k})")
plt.legend()
plt.show()
