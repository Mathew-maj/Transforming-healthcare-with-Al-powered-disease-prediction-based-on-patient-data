import streamlit as st
from model import predict_disease, load_model
import os

# Load model
model = load_model()

st.title("AI-Powered Disease Prediction")
st.write("Enter symptoms to get a prediction.")

age = st.slider("Age", 1, 100, 30)
fever = st.radio("Do you have a fever?", [1, 0], format_func=lambda x: "Yes" if x else "No")
cough = st.radio("Do you have a cough?", [1, 0], format_func=lambda x: "Yes" if x else "No")
headache = st.radio("Do you have a headache?", [1, 0], format_func=lambda x: "Yes" if x else "No")

if st.button("Predict Disease"):
    result = predict_disease(age, fever, cough, headache, model)
    st.success(f"Predicted Disease: {result}")

if os.path.exists("sentiment_plot.png"):
    st.image("sentiment_plot.png", caption="Prediction Distribution")