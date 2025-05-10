import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    df = pd.read_csv("sentiment140.csv")
    X = df.drop("disease", axis=1)
    y = df["disease"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")

def load_model():
    if not os.path.exists("model.pkl"):
        train_model()
    return joblib.load("model.pkl")

def predict_disease(age, fever, cough, headache, model):
    input_df = pd.DataFrame([[age, fever, cough, headache]],
                            columns=["age", "fever", "cough", "headache"])
    return model.predict(input_df)[0]