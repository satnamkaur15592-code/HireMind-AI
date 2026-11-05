import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="HireMind-AI | Placement Predictor", page_icon="🎓", layout="centered")

st.title("HireMind-AI - Placement Predictor")
st.write("Enter the student's academic details below. The AI model will predict whether the student is likely to be placed.")

@st.cache_data
def load_data():
    df = pd.read_csv("campus_placement_data.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_resource
def train_model(df):
    # FIXED: Target is 'placed' column, NOT last column (last is salary)
    target_column = "placed"
    # If placed not found, find any column with 'placed' in name
    if target_column not in df.columns:
        for c in df.columns:
            if "placed" in c.lower() or "status" in c.lower():
                target_column = c
                break

    y_raw = df[target_column].copy()

    # Features: Drop student_id, salary, and target
    drop_list = [target_column, "student_id", "salary"]
    X = df.drop(columns=[c for c in drop_list if c in df.columns]).copy()

    encoders = {}
    for col in X.columns:
        numeric_try = pd.to_numeric(X[col], errors="coerce")
        if numeric_try.notna().sum() > len(X) * 0.5:
            X[col] = numeric_try.fillna(numeric_try.mean())
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # Target: Placed / Not Placed -> 1 / 0
    target_encoder = None
    if y_raw.dtype == "object":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))
    else:
        y = y_raw

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, encoders, list(X.columns), target_encoder, X

data = load_data()
model, encoders, feature_columns, target_encoder, X_sample = train_model(data)

st.subheader("Enter Student Details")

user_input = {}
for column in feature_columns:
    if column in encoders:
        options = list(encoders[column].classes_)
        selected = st.selectbox(f"{column}", options)
        user_input[column] = encoders[column].transform([selected])[0]
    else:
        min_v = float(X_sample[column].min())
        max_v = float(X_sample[column].max())
        avg_v = float(X_sample[column].mean())
        user_input[column] = st.slider(f"{column}", min_value=min_v, max_value=max_v, value=avg_v)

if st.button("Predict Placement", use_container_width=True):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0].max() * 100
    if target_encoder is not None:
        result = target_encoder.inverse_transform([prediction])[0]
    else:
        result = prediction
    is_placed = str(result).lower() in ["1", "yes", "placed", "y"]
    if is_placed:
        st.success(f"✅ The student is likely to be placed. Confidence: {confidence:.1f}%")
    else:
        st.error(f"❌ The student is less likely to be placed. Confidence: {confidence:.1f}%")

st.divider()
st.caption("HireMind-AI | Built with Streamlit and Machine Learning")
