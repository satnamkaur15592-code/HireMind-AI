import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("HireMind-AI - Placement Predictor")
st.write("Enter the student marks, AI will predict if the student will be placed or not.")

# CSV se hi model bana lo, model.pkl ki zaroorat nahi
df = pd.read_csv("campus_placement_data.csv")

# Last wala column hi target hai (Placed / Placement)
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Sirf number wale columns rakho
X = X.select_dtypes(include=['number'])

model = RandomForestClassifier()
model.fit(X, y)

inputs = {}
for col in X.columns:
    inputs[col] = st.slider(col, 0, 100, 50)

if st.button("Predict"):
    data = pd.DataFrame([inputs])
    result = model.predict(data)[0]
    if result == 1:
        st.success("Placed - You will get placed!")
    else:
        st.error("Not Placed - You will not get placed")
