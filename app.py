import streamlit as st
import pandas as pd

st.title("HireMind-AI - Placement Predictor")
st.write("Debug Mode - Pehle hum CSV ke columns dekh rahe hai")

df = pd.read_csv("campus_placement_data.csv")
df.columns = [c.strip() for c in df.columns]

st.subheader("1. CSV me ye columns hai:")
st.write(list(df.columns))

st.subheader("2. Pehli 3 lines:")
st.write(df.head(3))

st.subheader("3. Har column me kitni alag values hai:")
for col in df.columns:
    st.write(f"{col} -> {df[col].nunique()} unique values, Example: {df[col].iloc[0]}")
