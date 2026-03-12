import streamlit as st
import pdfplumber
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="AI Resume Screening", layout="wide")

st.title("📄 AI Resume Screening & Candidate Ranking")
st.write("Upload multiple resumes and rank candidates based on match with job description.")

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ---------------- JOB DESCRIPTION ----------------
job_desc = st.text_area("Paste Job Description")

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload Multiple Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Analyze Candidates"):

    if not job_desc:
        st.warning("Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload at least one resume.")
        st.stop()

    cleaned_jd = clean_text(job_desc)

    results = []

    for file in uploaded_files:

        resume_text = ""

        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    resume_text += text

        cleaned_resume = clean_text(resume_text)

        # -------- Improved TF-IDF Similarity --------
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1,2),
            max_features=500
        )

        vectors = vectorizer.fit_transform([cleaned_resume, cleaned_jd])

        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        match_percentage = similarity * 100

        decision = "Shortlisted" if match_percentage >= 50 else "Not Shortlisted"

        results.append({
            "Candidate": file.name,
            "Match %": round(match_percentage,2),
            "Decision": decision
        })

    # ---------------- RANKING ----------------
    df_results = pd.DataFrame(results)

    df_results = df_results.sort_values(
        by="Match %",
        ascending=False
    ).reset_index(drop=True)

    df_results.index = df_results.index + 1

    st.subheader("🏆 Candidate Ranking")

    st.dataframe(df_results)

    # ---------------- TOP CANDIDATE ----------------
    top = df_results.iloc[0]

    st.success(
        f"Top Candidate: {top['Candidate']} with {top['Match %']}% match"
    )

    # ---------------- DOWNLOAD CSV ----------------
    csv = df_results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Ranking CSV",
        data=csv,
        file_name="candidate_ranking.csv",
        mime="text/csv"
    )
