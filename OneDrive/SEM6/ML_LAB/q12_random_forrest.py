from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Different numbers of trees
tree_counts = [10, 50, 100, 200]

print("Trees\tAccuracy\tPrecision\tRecall\t\tF1-Score")

for n in tree_counts:
    # Create Random Forest model
    rf = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    # Train model
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted'
    )
    recall = recall_score(
        y_test, y_pred, average='weighted'
    )
    f1 = f1_score(
        y_test, y_pred, average='weighted'
    )

    print(f"{n}\t{accuracy:.4f}\t\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}")
