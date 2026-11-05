import streamlit as st
import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("HireMind-AI - Placement Predictor")
st.write("Enter the student marks, AI will predict if the student will be placed or not.")

inputs = {}
for col in model.feature_names_in_:
    inputs[col] = st.slider(col, 0, 100, 50)

if st.button("Predict"):
    data = pd.DataFrame([inputs])
    result = model.predict(data)[0]
    if result == 1:
        st.success("Placed - You will get placed!")
    else:
        st.error("Not Placed - You will not get placed")