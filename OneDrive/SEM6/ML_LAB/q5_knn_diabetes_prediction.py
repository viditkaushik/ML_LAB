import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("datasets/breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CHANGE n_neighbors VALUE HERE
# Example: 3, 5, 7, 9

k_value = 5

# CHANGE DISTANCE METRIC HERE
# Possible values:
# 'euclidean'
# 'manhattan'
# 'minkowski'

distance_metric = 'euclidean'

model = KNeighborsClassifier(
    n_neighbors=k_value,
    metric=distance_metric
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("K Value:", k_value)

print("Distance Metric:", distance_metric)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_data = [X_test[0]]

prediction = model.predict(new_data)

print("\nNew Sample Prediction:")

if prediction[0] == 1:
    print("No Diabetes")
else:
    print("Diabetes")

plt.scatter(
    range(len(y_test)),
    y_test,
    label="Actual"
)

plt.scatter(
    range(len(y_pred)),
    y_pred,
    label="Predicted"
)

plt.title("KNN Prediction Graph")

plt.legend()

plt.show()