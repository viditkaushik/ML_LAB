import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


def load_iris_two_features_from_csv() -> np.ndarray:
    csv_path = os.path.join(os.path.dirname(__file__), "datasets", "iris.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != 'target']
        X = df[feature_cols].to_numpy()
        return X[:, :2]

    from sklearn.datasets import load_iris
    return load_iris().data[:, :2]


def main() -> None:
    X = load_iris_two_features_from_csv()

    model1 = DBSCAN(eps=0.3, min_samples=5)
    labels1 = model1.fit_predict(X)

    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels1)
    plt.title("DBSCAN eps=0.3 min_samples=5")
    plt.show()

    model2 = DBSCAN(eps=0.5, min_samples=3)
    labels2 = model2.fit_predict(X)

    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels2)
    plt.title("DBSCAN eps=0.5 min_samples=3")
    plt.show()


if __name__ == "__main__":
    main()

