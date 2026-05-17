import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# --- Data Loading & Preprocessing ---
df = pd.read_csv("datasets/diabetes1.csv")
df = df.dropna()  # preprocessing : removing rows with missing value
X = df.drop("Outcome", axis=1).values
y= df["Outcome"].values

scalar = MinMaxScaler()
X = scalar.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Custom KNN (Fully Vectorized) ---
def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        dists = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        k_labels = y_train[np.argsort(dists)[:k]].astype(int)
        preds.append(np.bincount(k_labels).argmax())
    return np.array(preds)

y_pred_custom = knn_predict(X_train, y_train, X_test, k=5)
print(f"Custom KNN (k=5) Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")

# --- Sklearn KNN with Tuning ---
ks = [3, 5, 7, 11, 15]
results = {}

for k in ks:
    for metric in ["euclidean", "manhattan"]:
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results[(k, metric)] = acc
        print(f"k={k:2d}, metric={metric:12s}, Accuracy={acc:.4f}")

# --- Plotting ---
accs = [results[(k, "euclidean")] for k in ks]
plt.figure(figsize=(6, 4))
plt.plot(ks, accs, "bo-")
plt.xlabel("K"); plt.ylabel("Accuracy"); plt.title("KNN: K vs Accuracy (Euclidean)")
plt.grid(True); plt.tight_layout(); plt.savefig("q5_knn.png", dpi=100); plt.show()