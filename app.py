import streamlit as st
import re
import io
import pandas as pd
from pypdf import PdfReader

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

st.title("AI Resume Screening Automation")
st.write("Analyze resumes against job requirements.")

job_title = st.text_input("Job Title")

job_description = st.text_area(
    "Enter the Job Description",
    height=180,
    placeholder="Enter the required skills and qualifications..."
)

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

def extract_text(file):
    reader = PdfReader(io.BytesIO(file.getvalue()))
    return " ".join(page.extract_text() or "" for page in reader.pages)

def extract_skills(text):
    skills = [
        "python", "java", "c++", "sql", "machine learning",
        "artificial intelligence", "html", "css", "javascript",
        "data analysis", "excel", "communication", "teamwork",
        "pandas", "scikit-learn", "deep learning"
    ]
    text = text.lower()
    return [skill for skill in skills if skill in text]

if st.button("Screen Resumes"):
    if not job_description.strip():
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF resume.")
    else:
        required_skills = extract_skills(job_description)
        results = []

        for file in uploaded_files:
            try:
                resume_text = extract_text(file)
                candidate_skills = extract_skills(resume_text)

                if required_skills:
                    matched = [
                        skill for skill in required_skills
                        if skill in candidate_skills
                    ]
                    score = round(
                        len(matched) / len(required_skills) * 100, 2
                    )
                else:
                    matched = []
                    score = 0

                results.append({
                    "Resume": file.name,
                    "Match Score (%)": score,
                    "Matched Skills": ", ".join(matched),
                    "Missing Skills": ", ".join(
                        skill for skill in required_skills
                        if skill not in candidate_skills
                    )
                })
            except Exception:
                st.warning(f"Could not read {file.name}.")
                
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values(
                "Match Score (%)", ascending=False
            ).reset_index(drop=True)

            st.subheader("Screening Results")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download Results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="resume_screening_results.csv",
                mime="text/csv"
            )

            st.caption(
                "Scores reflect keyword overlap with listed skills. "
                "Review resumes manually before making hiring decisions."
            )
