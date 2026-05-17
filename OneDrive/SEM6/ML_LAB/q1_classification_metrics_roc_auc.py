import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
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

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# MODEL
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("Confusion Matrix")
print(cm)

print("\nTP =", TP)
print("TN =", TN)
print("FP =", FP)
print("FN =", FN)

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def f1(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r)

def specificity(tn, fp):
    return tn / (tn + fp)

def npv(tn, fn):
    return tn / (tn + fn)

def mcc(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)

    denominator = np.sqrt(
        (tp + fp) *
        (tp + fn) *
        (tn + fp) *
        (tn + fn)
    )

    return numerator / denominator

print("\n--- OWN FUNCTION RESULTS ---")

print("Accuracy =", accuracy(TP, TN, FP, FN))
print("Precision =", precision(TP, FP))
print("Recall =", recall(TP, FN))
print("F1 Score =", f1(TP, FP, FN))
print("Specificity =", specificity(TN, FP))
print("NPV =", npv(TN, FN))
print("MCC =", mcc(TP, TN, FP, FN))

print("\n--- SKLEARN RESULTS ---")

print("Accuracy =", accuracy_score(y_test, y_pred))
print("Precision =", precision_score(y_test, y_pred))
print("Recall =", recall_score(y_test, y_pred))
print("F1 Score =", f1_score(y_test, y_pred))
print("MCC =", matthews_corrcoef(y_test, y_pred))

# SPECIFICITY
specificity_sklearn = TN / (TN + FP)
print("Specificity =", specificity_sklearn)

# NPV
npv_sklearn = TN / (TN + FN)
print("NPV =", npv_sklearn)

# -----------------------------
# ROC & AUC
# -----------------------------

# PREDICT PROBABILITIES
y_prob = model.predict_proba(X_test)[:, 1]

# RANDOM PROBABILITIES
random_probs = np.random.rand(len(y_test))

# ROC CURVE
fpr_model, tpr_model, _ = roc_curve(y_test, y_prob)
fpr_random, tpr_random, _ = roc_curve(y_test, random_probs)

# AUC
auc_model = auc(fpr_model, tpr_model)
auc_random = auc(fpr_random, tpr_random)

print("\nAUC OF MODEL =", auc_model)
print("AUC OF RANDOM PROBABILITIES =", auc_random)


# PLOT
plt.plot(fpr_model, tpr_model, label="Model AUC = %.3f" % auc_model)
plt.plot(fpr_random, tpr_random, label="Random AUC = %.3f" % auc_random)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.show()
# Plotting the Confusion Matrix (Doing it before this plt.show() causes image conflicts)
plt.imshow(cm)