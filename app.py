import streamlit as st
import joblib
import pdfplumber
import re
import os
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- NLTK SETUP (CLOUD SAFE) --------------------
nltk.data.path.append("./nltk_data")

if not os.path.exists("nltk_data"):
    nltk.download("stopwords", download_dir="nltk_data")

stop_words = set(stopwords.words("english"))

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    layout="centered"
)

st.title("📄 AI Resume Screening System")
st.write("Upload a resume PDF and get instant screening results")

# -------------------- HELPER FUNCTIONS --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def extract_skills(text):
    skills = [
        "python", "machine learning", "data science", "sql",
        "classification", "anomaly detection",
        "pandas", "numpy", "scikit learn",
        "model training", "data analysis",
        "tensorflow", "flask", "streamlit"
    ]
    return [skill for skill in skills if skill in text]

# -------------------- LOAD MODEL --------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resume_model.pkl")
model = joblib.load(MODEL_PATH)

# -------------------- USER INPUTS --------------------
st.subheader("📋 Paste Job Description")
job_desc = st.text_area("Enter job description here")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# -------------------- PREDICTION & OUTPUT --------------------
if uploaded_file is not None and job_desc.strip() != "":
    with pdfplumber.open(uploaded_file) as pdf:
        resume_text = ""
        for page in pdf.pages:
            if page.extract_text():
                resume_text += page.extract_text()

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_desc)

    if st.button("Predict"):
        # ---- CATEGORY PREDICTION ----
        prediction = model.predict([cleaned_resume])[0]
        st.success(f"✅ Resume Category: {prediction}")

        # ---- JOB MATCH PERCENTAGE (MAIN SCORE) ----
        vectors = model.named_steps["tfidf"].transform(
            [cleaned_resume, cleaned_jd]
        )
        match_percentage = cosine_similarity(
            vectors[0], vectors[1]
        )[0][0] * 100

        st.metric(
            label="📊 Match Percentage",
            value=f"{match_percentage:.2f} %"
        )

        st.progress(int(match_percentage))

        # ---- SHORTLIST DECISION ----
        if match_percentage >= 60:
            st.success("🎯 Decision: SHORTLISTED")
        else:
            st.warning("⚠️ Decision: NEEDS MANUAL REVIEW")

        # ---- SKILL EXTRACTION ----
        st.subheader("🛠️ Detected Skills")
        skills_found = extract_skills(cleaned_resume)

        if skills_found:
            st.write(", ".join(skills_found))
        else:
            st.write("No major ML-related skills detected")

        # ---- EXPLANATION NOTE ----
        st.info(
            "ℹ️ Match Percentage represents similarity between resume and job description "
            "using TF-IDF and cosine similarity. Higher values indicate stronger relevance."
        )

elif uploaded_file is not None:
    st.warning("⚠️ Please paste a Job Description to calculate match percentage.")
