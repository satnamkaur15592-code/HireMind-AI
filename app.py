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
st.write(
    "Enter the student's academic details below. "
    "The AI model will predict whether the student is likely to be placed."
)

@st.cache_data
def load_data():
    df = pd.read_csv("campus_placement_data.csv")
    df.columns = [col.strip() for col in df.columns]
    return df

@st.cache_resource
def train_model(df):
    target_column = df.columns[-1]

    # Drop ID / Name columns - they are not useful for prediction
    # Example: Student ID like STU001, Name, etc.
    drop_cols = []
    for col in df.columns:
        low = col.lower()
        if "id" in low or "name" in low:
            drop_cols.append(col)

    df_clean = df.drop(columns=drop_cols, errors="ignore")
    feature_columns = [c for c in df_clean.columns if c != target_column]

    X = df_clean[feature_columns].copy()
    y = df_clean[target_column].copy()

    # Encode categorical features
    encoders = {}
    for column in X.columns:
        if X[column].dtype == "object":
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))
            encoders[column] = encoder
        else:
            # Force numeric columns to numbers
            X[column] = pd.to_numeric(X[column], errors="coerce")
            X[column] = X[column].fillna(X[column].mean())

    # Encode target
    target_encoder = None
    if y.dtype == "object":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, encoders, feature_columns, target_encoder, df_clean

data = load_data()
model, encoders, feature_columns, target_encoder, df_clean = train_model(data)

st.subheader("Enter Student Details")

user_input = {}
for column in feature_columns:
    if column in encoders:
        options = list(encoders[column].classes_)
        selected_value = st.selectbox(f"{column}", options)
        user_input[column] = encoders[column].transform([selected_value])[0]
    else:
        min_value = float(df_clean[column].min())
        max_value = float(df_clean[column].max())
        average_value = float(df_clean[column].mean())
        user_input[column] = st.slider(
            f"{column}",
            min_value=min_value,
            max_value=max_value,
            value=average_value
        )

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
