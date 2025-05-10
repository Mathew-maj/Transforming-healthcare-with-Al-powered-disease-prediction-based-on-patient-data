import streamlit as st
from main import predict_disease, load_model
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load trained model
model = load_model()

# App title
st.title("AI-Powered Disease Prediction")
st.write("Enter patient symptoms to predict the disease.")

# Input fields
age = st.slider("Age", 1, 100, 25)
fever = st.radio("Do you have a fever?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
cough = st.radio("Do you have a cough?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
headache = st.radio("Do you have a headache?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Predict button
if st.button("Predict Disease"):
    prediction = predict_disease(age, fever, cough, headache, model)
    st.success(f"Predicted Disease: {prediction}")

# Display histogram image if available
if os.path.exists("sentiment_plot.png"):
    st.markdown("### Disease Prediction Distribution")
    st.image("sentiment_plot.png", use_column_width=True)