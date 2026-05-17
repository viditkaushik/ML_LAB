import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
    roc_curve,
    auc
)

df = pd.read_csv("breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_pred_binary = [1 if x > 0.5 else 0 for x in linear_pred]

print("\nLINEAR REGRESSION")

print("Accuracy:",accuracy_score(y_test, linear_pred_binary))
print("Precision:",precision_score(y_test, linear_pred_binary))
print("Recall:",recall_score(y_test, linear_pred_binary))

mse_linear = mean_squared_error(y_test, linear_pred)

print("MSE:", mse_linear)
print("RMSE:", np.sqrt(mse_linear))

cm_linear = confusion_matrix(y_test, linear_pred_binary)

print("Confusion Matrix:")
print(cm_linear)

logistic_model = LogisticRegression(max_iter=5000)
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)

print("\nLOGISTIC REGRESSION")

print("Accuracy:",accuracy_score(y_test, logistic_pred))
print("Precision:",precision_score(y_test, logistic_pred))
print("Recall:",recall_score(y_test, logistic_pred))

mse_log = mean_squared_error(y_test, logistic_pred)

print("MSE:", mse_log)
print("RMSE:", np.sqrt(mse_log))
cm_log = confusion_matrix(y_test, logistic_pred)

print("Confusion Matrix:")
print(cm_log)

new_data = [X_test.iloc[0]]

print("\nNEW DATA PREDICTION")

print("Linear Regression:",linear_model.predict(new_data))
print("Logistic Regression:",logistic_model.predict(new_data))

plt.plot(y_test.values[:30], label="Actual")
plt.plot(linear_pred[:30], label="Predicted")

plt.title("Linear Regression")
plt.legend()
plt.show()

log_probs = logistic_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, log_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.show()
plt.imshow(cm_log)

plt.title("Confusion Matrix")
plt.colorbar()
plt.show()