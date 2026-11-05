import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="HireMind-AI | Placement Predictor", page_icon="🎓", layout="centered")
st.title("HireMind-AI - Placement Journey")
st.write("A 3-stage placement process - Just like a real company hiring process.")

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

if "step" not in st.session_state:
    st.session_state.step = 1
if "resume_passed" not in st.session_state:
    st.session_state.resume_passed = False
if "interview_passed" not in st.session_state:
    st.session_state.interview_passed = False
if "resume_data" not in st.session_state:
    st.session_state.resume_data = {}
if "interview_data" not in st.session_state:
    st.session_state.interview_data = {}

st.progress(st.session_state.step / 3)
st.write(f"**Step {st.session_state.step} / 3**")

# ================= ROUND 1 =================
if st.session_state.step == 1:
    st.header("📄 ROUND 1: Resume Shortlisting")
    st.write("The company will first review your resume - Academic Scores, Internships, Projects and Certifications.")

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
    st.subheader("Additional Resume Details:")
    internship = st.slider("Internship Count", 0, 5, 1)
    project = st.slider("Project Count", 0, 10, 2)
    cert = st.slider("Certification Count", 0, 10, 2)

    if st.button("Check Resume Shortlist", use_container_width=True):
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
            st.session_state.resume_passed = True
        else:
            st.session_state.resume_passed = False

    # Show result separately - NOT inside button, so Next button will work
    if "conf" in st.session_state.resume_data:
        r = st.session_state.resume_data
        if st.session_state.resume_passed:
            st.success(f"✅ Congratulations! Your Resume is SHORTLISTED! (Model Confidence: {r['conf']:.1f}% | Resume Score: {r['resume_score']})")
            st.write("You are eligible for Round 2 - Interview Round.")
        else:
            st.error(f"❌ Sorry! Your Resume is NOT SHORTLISTED. (Model Confidence: {r['conf']:.1f}% | Resume Score: {r['resume_score']})")
            st.write("Tip: Improve Internship / Projects / Certifications and try again.")

    if st.session_state.resume_passed:
        if st.button("Next -> Round 2 Interview", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

# ================= ROUND 2 =================
elif st.session_state.step == 2:
    st.header("🎤 ROUND 2: Interview Round")
    st.write("Resume shortlisted! Now the Interview Round will be conducted.")

    comm = st.slider("Communication Skills (1-10)", 1, 10, 6)
    soft = st.slider("Soft Skills / Confidence (1-10)", 1, 10, 6)
    tech_skill = st.slider("Technical Skill (1-10)", 1, 10, 6)
    aptitude = st.slider("Aptitude / Coding Test Score (0-100)", 0, 100, 60)
    tech_interview = st.selectbox("Technical Interview Result", ["Cleared", "Not Cleared"])
    hr_interview = st.selectbox("HR Interview Result", ["Cleared", "Not Cleared"])

    if st.button("Check Interview Result", use_container_width=True):
        interview_clear = (tech_interview == "Cleared" and hr_interview == "Cleared")
        avg_skill = (comm + soft + tech_skill) / 3
        st.session_state.interview_data = {"clear": interview_clear, "avg": avg_skill, "aptitude": aptitude, "comm": comm}
        if interview_clear and avg_skill >= 5:
            st.session_state.interview_passed = True
        else:
            st.session_state.interview_passed = False

    if "avg" in st.session_state.interview_data:
        i = st.session_state.interview_data
        if st.session_state.interview_passed:
            st.success(f"✅ Interview CLEARED! (Average Skill: {i['avg']:.1f}/10)")
        else:
            st.error(f"❌ Interview NOT CLEARED. Average Skill: {i['avg']:.1f}/10")
            st.write("Please improve Communication / Technical Skills.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Round 1"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.session_state.interview_passed:
            if st.button("Next -> Final Result"):
                st.session_state.step = 3
                st.rerun()

# ================= ROUND 3 =================
elif st.session_state.step == 3:
    st.header("🏆 ROUND 3: Final Placement Result")
    r = st.session_state.get("resume_data", {})
    i = st.session_state.get("interview_data", {})

    st.write(f"📄 **Round 1 Resume:** Shortlisted (Confidence: {r.get('conf',0):.1f}%)")
    st.write(f"🎤 **Round 2 Interview:** {'Cleared' if i.get('clear') else 'Not Cleared'} (Average Skill: {i.get('avg',0):.1f}/10 | Aptitude: {i.get('aptitude',0)})")

    st.divider()
    if r.get("model_placed") and i.get("clear"):
        st.success("🎉 FINAL RESULT: PLACED! Congratulations, you got the job!")
        st.balloons()
    elif i.get("clear") and not r.get("model_placed"):
        st.warning("⚠️ FINAL RESULT: BORDERLINE PLACED - Interview is cleared, but academic score is slightly low. Company is still considering.")
    else:
        st.error("❌ FINAL RESULT: NOT PLACED - One of the rounds was not cleared.")

    if st.button("🔄 Check for New Student - Start Again"):
        st.session_state.step = 1
        st.session_state.resume_passed = False
        st.session_state.interview_passed = False
        st.session_state.resume_data = {}
        st.session_state.interview_data = {}
        st.rerun()

st.divider()
st.caption("HireMind-AI | Round 1 Resume -> Round 2 Interview -> Round 3 Final Placement")
