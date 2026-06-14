import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# 1. Load dataset from CSV
# Replace with your actual CSV path
df = pd.read_csv("datasets/breast_cancer.csv")

# Assume the target column is named "Outcome" (adjust if different)
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train models
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)

ada.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# 3. Evaluate
for name, model in [("AdaBoost", ada), ("XGBoost", xgb)]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{'='*40}\n{name}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))
