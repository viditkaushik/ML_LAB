import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix



data = {
    'Weather': ['Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny'],
    'Temperature': ['Hot', 'Cool', 'Mild', 'Mild', 'Cool', 'Hot'],
    'Play': ['No', 'Yes', 'Yes', 'Yes', 'Yes', 'No'],
}

df = pd.DataFrame(data)
print("DATASET")
print(df)

le = LabelEncoder()
df['Weather'] = le.fit_transform(df['Weather'])
df['Temperature'] = le.fit_transform(df['Play'])  
df['Play'] = le.fit_transform(df['Play'])

print(df)

X = df[['Weather', 'Temperature']]
Y = df['Play']

model = GaussianNB()
model.fit(X, Y)

Y_PRED = model.predict(X)

print("Accuracy", accuracy_score(Y, Y_PRED), "\n")
print("Confusion Matrix \n", confusion_matrix(Y, Y_PRED))

test = [[1, 1]]
pred = model.predict(test)

if pred == 1:
    print("Yes")
else:
    print("No")

