import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="HireMind-AI | Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("HireMind-AI - Placement Predictor")
st.write("Enter the student's academic details below. The AI model will predict whether the student is likely to be placed.")

@st.cache_data
def load_data():
    df = pd.read_csv("campus_placement_data.csv")
    df.columns = [col.strip() for col in df.columns]
    return df

@st.cache_resource
def train_model(df):
    target_column = df.columns[-1]
    y_raw = df[target_column].copy()

    # Features = all columns except last one
    X = df.drop(columns=[target_column]).copy()

    # STEP 1: Drop ID-like columns - Student ID, Roll No, anything with too many unique values
    cols_to_drop = []
    for col in X.columns:
        low = col.lower()
        # Name / ID in column name
        if "id" in low or "name" in low or "roll" in low or "student" in low and "id" in low:
            cols_to_drop.append(col)
        # OR if column has STU001 type values - too many unique text values = ID
        elif X[col].dtype == "object" and X[col].nunique() > 30:
            cols_to_drop.append(col)

    X = X.drop(columns=cols_to_drop, errors="ignore")

    # STEP 2: Convert EVERY column to number, forcefully
    encoders = {}
    for col in X.columns:
        # Try to make it numeric first
        numeric_try = pd.to_numeric(X[col], errors="coerce")
        # If most values became numbers, use numeric
        if numeric_try.notna().sum() > len(X) * 0.5:
            X[col] = numeric_try.fillna(numeric_try.mean())
        else:
            # Otherwise it's text like Male/Female, Yes/No - encode it
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # STEP 3: Target also to number
    target_encoder = None
    if y_raw.dtype == "object":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0)

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
        # If CGPA type small range, slider fine
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
