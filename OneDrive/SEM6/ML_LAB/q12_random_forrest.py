import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("datasets/diabetes1.csv").dropna()
X= df.drop("Outcome", axis=1) 
y=df["Outcome"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_trees_list = [1, 5, 10, 20, 50, 100, 150, 200]
results = []
for n in n_trees_list:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results.append({
        "n_trees": n,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

for metric in ["accuracy","precision","recall","f1"]:
    plt.plot(res_df["n_trees"], res_df[metric], marker="o", label=metric.capitalize())
plt.xlabel("Number of Trees"); plt.ylabel("Score")
plt.title("Random Forest: Effect of Number of Trees")
plt.show()
