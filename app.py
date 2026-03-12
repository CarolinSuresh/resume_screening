import streamlit as st
import pdfplumber
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="AI Resume Screening", layout="wide")

st.title("📄 AI Resume Screening & Candidate Ranking")
st.write("Upload multiple resumes and rank candidates based on their match with the job description.")

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ---------------- JOB DESCRIPTION INPUT ----------------
job_desc = st.text_area("Paste Job Description")

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload Multiple Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------- ANALYZE BUTTON ----------------
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

        # Extract text from PDF
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    resume_text += text

        cleaned_resume = clean_text(resume_text)

        # -------- TF-IDF Similarity --------
        vectorizer = TfidfVectorizer(stop_words="english")

        docs = [cleaned_jd, cleaned_resume]
        tfidf_matrix = vectorizer.fit_transform(docs)

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # -------- Keyword Overlap Score --------
        jd_words = set(cleaned_jd.split())
        resume_words = set(cleaned_resume.split())

        common_keywords = jd_words.intersection(resume_words)
        keyword_score = len(common_keywords) / max(len(jd_words), 1)

        # -------- Final Combined Score --------
        final_score = (0.7 * similarity) + (0.3 * keyword_score)

        match_percentage = final_score * 100

        decision = "Shortlisted" if match_percentage >= 50 else "Not Shortlisted"

        results.append({
            "Candidate": file.name,
            "Match %": round(match_percentage, 2),
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

    st.dataframe(df_results, use_container_width=True)

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
