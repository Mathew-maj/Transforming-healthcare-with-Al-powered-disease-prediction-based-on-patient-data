import pandas as pd
from main import load_model
from sklearn.metrics import classification_report

df = pd.read_csv('data.csv')
X = df.drop('disease', axis=1)
y = df['disease']
model = load_model()
predictions = model.predict(X)

print(classification_report(y, predictions))