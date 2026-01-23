# AI-Based Resume Screening System

## Project Overview
The AI-Based Resume Screening System is a machine learning application that automates the initial screening of resumes.  
It helps recruiters and HR professionals analyze resumes efficiently by extracting relevant skills, matching them with job descriptions, and providing a preliminary shortlisting decision.

The system uses Machine Learning and Natural Language Processing (NLP) techniques and is deployed as a web application using Streamlit.

---

## Objectives
- Automate resume screening using machine learning
- Reduce manual effort and bias in resume shortlisting
- Classify resumes into relevant job categories
- Extract key technical skills from resumes
- Match resumes with job descriptions
- Provide a shortlisting decision with confidence score

---

## Technologies Used
- Programming Language: Python
- Machine Learning: TF-IDF, Naive Bayes
- Libraries:
  - scikit-learn
  - pandas
  - nltk
  - joblib
  - pdfplumber
- Web Framework: Streamlit
- Deployment: Streamlit Cloud

---

## Project Structure
resume-screening-ml/
│
├── app.py                 # Streamlit web application
├── resume_model.pkl       # Trained machine learning model
├── requirements.txt       # Required Python libraries
├── README.md              # Project documentation

---

## Features
- Upload resume in PDF format
- Resume category prediction using ML
- Confidence score and resume score display
- Automatic skill extraction
- Job description matching using cosine similarity
- Shortlisting decision (Shortlisted / Needs Review)
- Simple and user-friendly web interface

---

## Machine Learning Approach
- Text Preprocessing:
  - Lowercasing
  - Stopword removal
  - Text cleaning
- Feature Extraction:
  - TF-IDF Vectorization
- Classification Model:
  - Multinomial Naive Bayes
- Similarity Measure:
  - Cosine Similarity for job description matching

---

## Deployment
The application is deployed using Streamlit Cloud and is accessible through a web browser.  
For academic demonstration, the project can also be executed locally using Streamlit.

---

## How to Run Locally
1. Clone the repository:
   git clone https://github.com/your-username/resume-screening-ml.git

2. Navigate to the project directory:
   cd resume-screening-ml

3. Install required dependencies:
   pip install -r requirements.txt

4. Run the application:
   python -m streamlit run app.py

---

## Use Cases
- Resume screening for HR teams
- Academic machine learning project
- NLP-based text classification learning
- Prototype for Applicant Tracking Systems (ATS)

---

## Future Enhancements
- Skill-based resume ranking
- Bias reduction mechanisms
- Support for multiple job roles
- Advanced NLP models such as BERT
- Database integration for resume storage

---

## Disclaimer
This project is developed for academic and educational purposes and demonstrates the use of machine learning techniques in resume screening.

---

## Author
Carolin Suresh  
Electronics and Communication Engineering  
Machine Learning Minor Project
