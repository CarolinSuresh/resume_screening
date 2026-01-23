import streamlit as st
import joblib
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="AI Resume Screening System")

st.title("📄 AI Resume Screening System")
st.write("Upload a resume PDF and get instant screening results")

model = joblib.load("resume_model.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    cleaned = clean_text(text)

    if st.button("Predict"):
        prediction = model.predict([cleaned])[0]
        probability = max(model.predict_proba([cleaned])[0]) * 100

        st.success(f"✅ Resume Category: {prediction}")
        st.info(f"📊 Confidence Score: {probability:.2f}%")
