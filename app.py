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
        "deep learning", "pandas", "numpy", "tensorflow",
        "flask", "streamlit", "java", "c", "c++", "excel",
        "html", "css", "javascript"
    ]
    return [skill for skill in skills if skill in text]

# -------------------- LOAD MODEL --------------------
model = joblib.load("resume_model.pkl")

# -------------------- USER INPUTS --------------------
st.subheader("📋 Optional: Paste Job Description")
job_desc = st.text_area("Enter job description here")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# -------------------- PREDICTION & OUTPUT --------------------
if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        resume_text = ""
        for page in pdf.pages:
            if page.extract_text():
                resume_text += page.extract_text()

    cleaned_resume = clean_text(resume_text)

    if st.button("Predict"):
        prediction = model.predict([cleaned_resume])[0]
        probability = max(model.predict_proba([cleaned_resume])[0]) * 100

        # -------- MAIN RESULTS --------
        st.success(f"✅ Resume Category: {prediction}")
        st.info(f"📊 Confidence Score: {probability:.2f}%")
        st.metric("📈 Resume Score", f"{probability:.2f}/100")

        # -------- SHORTLIST DECISION --------
        if probability >= 60:
            st.success("🎯 Decision: SHORTLISTED")
        else:
            st.warning("⚠️ Decision: NEEDS MANUAL REVIEW")

        # -------- SKILL EXTRACTION --------
        st.subheader("🛠️ Detected Skills")
        skills_found = extract_skills(cleaned_resume)

        if skills_found:
            st.write(", ".join(skills_found))
        else:
            st.write("No major skills detected")

        # -------- JOB DESCRIPTION MATCHING --------
        if job_desc.strip() != "":
            cleaned_jd = clean_text(job_desc)
            vectors = model.named_steps["tfidf"].transform(
                [cleaned_resume, cleaned_jd]
            )
            similarity = cosine_similarity(
                vectors[0], vectors[1]
            )[0][0] * 100

            st.subheader("📊 Job Match Score")
            st.progress(int(similarity))
            st.write(f"Match Percentage: {similarity:.2f}%")

        # -------- LOW CONFIDENCE EXPLANATION --------
        if probability < 40:
            st.info(
                "ℹ️ Low confidence may be due to limited matching keywords or resume format."
            )
