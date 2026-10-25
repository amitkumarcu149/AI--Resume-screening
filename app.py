import streamlit as st
import pandas as pd
import re
import io
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

st.title("AI-Based Resume Screening Automation")
st.write("Compare candidate resumes with job requirements using NLP and TF-IDF.")

job_title = st.text_input("Job Title")

job_description = st.text_area(
    "Enter Job Description",
    height=180,
    placeholder="Enter the job role, required skills and qualifications..."
)

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)


def extract_text(file):
    reader = PdfReader(io.BytesIO(file.getvalue()))
    return " ".join(page.extract_text() or "" for page in reader.pages)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


if st.button("Screen Resumes"):
    if not job_description.strip():
        st.error("Please enter a job description.")

    elif not uploaded_files:
        st.error("Please upload at least one PDF resume.")

    else:
        try:
            resume_texts = []
            valid_files = []

            for file in uploaded_files:
                try:
                    text = extract_text(file)
                    if text.strip():
                        resume_texts.append(clean_text(text))
                        valid_files.append(file.name)
                    else:
                        st.warning(
                            f"No readable text found in {file.name}."
                        )
                except Exception:
                    st.warning(f"Could not read {file.name}.")

            if not resume_texts:
                st.error("No readable resume text was found.")

            else:
                job_text = clean_text(job_description)

                documents = [job_text] + resume_texts

                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=10000
                )

                matrix = vectorizer.fit_transform(documents)

                scores = cosine_similarity(
                    matrix[0:1],
                    matrix[1:]
                ).flatten() * 100

                results = []

                for filename, resume, score in zip(
                    valid_files, resume_texts, scores
                ):
                    results.append({
                        "Resume": filename,
                        "Match Score (%)": round(float(score), 2),
                        "Screening Suggestion": (
                            "Potential Match"
                            if score >= 30
                            else "Needs Manual Review"
                        )
                    })

                df = pd.DataFrame(results)
                df = df.sort_values(
                    "Match Score (%)",
                    ascending=False
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
                    "This prototype uses TF-IDF and cosine similarity. "
                    "The score is a text-similarity measure, not a "
                    "trained SVM prediction or an accuracy score. "
                    "Review every resume manually before hiring decisions."
                )

        except Exception as error:
            st.error(f"An error occurred: {error}")
