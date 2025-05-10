import streamlit as st
from main import predict_disease, load_model

st.title("AI Disease Prediction")

age = st.slider("Age", 1, 100, 25)
fever = st.radio("Fever", [0, 1])
cough = st.radio("Cough", [0, 1])
headache = st.radio("Headache", [0, 1])

if st.button("Predict"):
    model = load_model()
    result = predict_disease(age, fever, cough, headache, model)
    st.success(f"Predicted Disease: {result}")