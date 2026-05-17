import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score



data = {
    'Weather': ['Sunny', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Rainy'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Strong', 'Weak', 'Weak'],
    'Play': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'],
}

df = pd.DataFrame(data)
print("DATASET")
print(df)

le = LabelEncoder()
df['Weather'] = le.fit_transform(df['Weather'])
df['Wind'] = le.fit_transform(df['Wind'])
df['Play'] = le.fit_transform(df['Play'])

X = df[['Weather', 'Wind']]
y = df['Play']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

y_pred = model.predict(X)

print("\nAccuracy:")
print(accuracy_score(y, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

# new sample: Sunny + Weak
# LabelEncoder sorts alphabetically: Rainy -> 0 Sunny -> 1
# similarly: Strong -> 0 Weak -> 1
new_data = [[1, 1]]
prediction = model.predict(new_data)

print("\nNew Sample Prediction:")
if prediction[0] == 1:
    print("Play = Yes")
else:
    print("Play = No")

plot_tree(
    model,
    feature_names=['Weather', 'Wind'],
    class_names=['No', 'Yes'],
    filled=True,
)
plt.show()


