import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    auc
)

df = pd.read_csv("breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CHANGE max_iter VALUE HERE
# Example: 1000, 3000, 5000

max_iterations = 5000

model = LogisticRegression(
    max_iter=max_iterations
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("Max Iterations:", max_iterations)

print("\nConfusion Matrix")
print(cm)

print("\nTP =", TP)
print("TN =", TN)
print("FP =", FP)
print("FN =", FN)

print("\nAccuracy =", accuracy_score(y_test, y_pred))

print("Precision =",precision_score(y_test, y_pred))

print("Recall =",recall_score(y_test, y_pred))

print("F1 Score =",f1_score(y_test, y_pred))

print("MCC =",matthews_corrcoef(y_test, y_pred))

specificity = TN / (TN + FP)

print("Specificity =",specificity)

npv = TN / (TN + FN)

print("NPV =",npv)

y_prob = model.predict_proba(X_test)[:,1]
random_probs = np.random.rand(len(y_test))

fpr_model, tpr_model, _ = roc_curve(y_test, y_prob)
fpr_random, tpr_random, _ = roc_curve(y_test,random_probs)

auc_model = auc(fpr_model, tpr_model)
auc_random = auc(fpr_random, tpr_random)

print("\nAUC OF MODEL =", auc_model)
print("AUC OF RANDOM =", auc_random)

plt.plot(fpr_model,tpr_model,label="Model AUC = %.3f" % auc_model)

plt.plot(fpr_random,tpr_random,label="Random AUC = %.3f" % auc_random)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()