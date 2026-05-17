import pandas as pd

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Yes'],
]

columns = ['Sky', 'Temp', 'Humidity', 'Wind', 'EnjoySport']
df = pd.DataFrame(data, columns=columns)

print(df)

hypothesis = [0] * (len(columns) - 1)

for row in data:
    if row[-1] == 'Yes':
        for i in range(len(hypothesis)):
            if hypothesis[i] == 0:
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = '?'

print("Most Specific Hypothesis")
print(hypothesis)