import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="HireMind-AI | Placement Predictor", page_icon="🎓", layout="centered")
st.title("HireMind-AI - Placement Journey")
st.write("Jaise asli company me hota hai - 3 Round me placement hoga.")

@st.cache_data
def load_data():
    df = pd.read_csv("campus_placement_data.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_resource
def train_model(df):
    target_column = "placed"
    if target_column not in df.columns:
        for c in df.columns:
            if "placed" in c.lower():
                target_column = c
                break
    y_raw = df[target_column].copy()
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

# Step ko yaad rakhne ke liye
if "step" not in st.session_state:
    st.session_state.step = 1
if "resume_data" not in st.session_state:
    st.session_state.resume_data = {}

# Progress bar
st.progress(st.session_state.step / 3)
st.write(f"**Step {st.session_state.step} / 3**")

# ================= ROUND 1 =================
if st.session_state.step == 1:
    st.header("📄 ROUND 1: Resume Shortlisting")
    st.write("Pehle company tera Resume dekhegi - Marks + Internship + Projects + Certificates")

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

    st.divider()
    st.subheader("Resume ke extra points:")
    internship = st.slider("Internship Count", 0, 5, 1)
    project = st.slider("Project Count", 0, 10, 2)
    cert = st.slider("Certification Count", 0, 10, 2)

    if st.button("Check Resume Shortlist ->", use_container_width=True):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        conf = model.predict_proba(input_df)[0].max() * 100
        if target_encoder is not None:
            result = target_encoder.inverse_transform([pred])[0]
        else:
            result = pred
        model_placed = str(result).lower() in ["1", "yes", "placed", "y"]
        resume_score = internship*10 + project*5 + cert*5

        st.session_state.resume_data = {"model_placed": model_placed, "conf": conf, "resume_score": resume_score}

        if model_placed or resume_score >= 30:
            st.success(f"✅ CONGRATS! Resume SHORTLIST ho gaya! (Model: {conf:.1f}% | Resume Score: {resume_score})")
            st.write("Ab tu Round 2 - Interview me ja sakti hai.")
            if st.button("Next -> Round 2 Interview"):
                st.session_state.step = 2
                st.rerun()
        else:
            st.error(f"❌ SORRY! Resume SHORTLIST nahi hua. (Model: {conf:.1f}% | Resume Score: {resume_score})")
            st.write("Tip: Internship / Project / Certificate badhao, fir try karo.")

# ================= ROUND 2 =================
elif st.session_state.step == 2:
    st.header("🎤 ROUND 2: Interview Round")
    st.write("Resume clear ho gaya! Ab Interview hoga.")

    comm = st.slider("Communication Skills (1-10)", 1, 10, 6)
    soft = st.slider("Soft Skills / Confidence (1-10)", 1, 10, 6)
    tech_skill = st.slider("Technical Skill (1-10)", 1, 10, 6)
    aptitude = st.slider("Aptitude / Coding Test Score (0-100)", 0, 100, 60)
    tech_interview = st.selectbox("Technical Interview kaisa gaya?", ["Clear Ho Gaya", "Clear Nahi Hua"])
    hr_interview = st.selectbox("HR Interview kaisa gaya?", ["Clear Ho Gaya", "Clear Nahi Hua"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("<- Back to Round 1"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Check Interview Result ->", use_container_width=True):
            interview_clear = (tech_interview == "Clear Ho Gaya" and hr_interview == "Clear Ho Gaya")
            avg_skill = (comm + soft + tech_skill) / 3
            st.session_state.interview_data = {"clear": interview_clear, "avg": avg_skill, "aptitude": aptitude, "comm": comm}
            if interview_clear and avg_skill >= 5:
                st.success(f"✅ Interview CLEAR! (Avg Skill: {avg_skill:.1f}/10)")
                if st.button("Next -> Final Result"):
                    st.session_state.step = 3
                    st.rerun()
            else:
                st.error(f"❌ Interview CLEAR nahi hua. Avg Skill: {avg_skill:.1f}/10")
                st.write("Communication / Technical skill improve karo.")

# ================= ROUND 3 =================
elif st.session_state.step == 3:
    st.header("🏆 ROUND 3: Final Placement Result")
    r = st.session_state.get("resume_data", {})
    i = st.session_state.get("interview_data", {})

    st.write(f"📄 **Round 1 Resume:** Shortlisted (Confidence: {r.get('conf',0):.1f}%)")
    st.write(f"🎤 **Round 2 Interview:** {'Clear' if i.get('clear') else 'Not Clear'} (Avg Skill: {i.get('avg',0):.1f}/10 | Aptitude: {i.get('aptitude',0)})")

    st.divider()
    if r.get("model_placed") and i.get("clear"):
        st.success("🎉 FINAL RESULT: PLACED! Tumhe job mil gayi!")
        st.balloons()
    elif i.get("clear") and not r.get("model_placed"):
        st.warning("⚠️ FINAL RESULT: BORDERLINE PLACED - Interview clear hai, par academic thoda weak tha. Company soch rahi hai.")
    else:
        st.error("❌ FINAL RESULT: NOT PLACED - Koi ek round me atak gaye.")

    if st.button("🔄 Naya Student Check Karo - Start Again"):
        st.session_state.step = 1
        st.rerun()

st.divider()
st.caption("HireMind-AI | Round 1 Resume -> Round 2 Interview -> Round 3 Final Placement")
