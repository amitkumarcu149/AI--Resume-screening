import streamlit as st
import pandas as pd
import io
import joblib
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

st.title("AI-Based Resume Screening Automation")
st.write("Screen resumes using an NLP model and job-description similarity.")

@st.cache_resource
def load_model():
    saved = joblib.load("resume_svm_model.pkl")
    return saved["model"], saved["vectorizer"]

try:
    model, vectorizer = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error("Could not load the trained model. Check that resume_svm_model.pkl is in the repository.")
    st.caption(str(e))

job_title = st.text_input("Job Title")

job_description = st.text_area(
    "Enter Job Description",
    height=180,
    placeholder="Enter the job role, skills and qualifications..."
)

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

def extract_text(file):
    reader = PdfReader(io.BytesIO(file.getvalue()))
    return " ".join(page.extract_text() or "" for page in reader.pages)

if st.button("Screen Resumes"):
    if not job_description.strip():
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one PDF resume.")
    elif not model_ready:
        st.error("The trained model could not be loaded.")
    else:
        results = []
        readable_resumes = []

        for file in uploaded_files:
            try:
                text = extract_text(file)
                if text.strip():
                    readable_resumes.append((file.name, text))
                else:
                    st.warning(f"No readable text found in {file.name}.")
            except Exception:
                st.warning(f"Could not read {file.name}.")

        if readable_resumes:
            texts = [text for _, text in readable_resumes]

            # Predict the resume's job category using the trained SVM.
            resume_vectors = vectorizer.transform(texts)
            predictions = model.predict(resume_vectors)

            # Calculate similarity to the entered job description.
            # This is a separate similarity score, not SVM confidence.
            job_vector = vectorizer.transform([job_description])
            similarity_scores = cosine_similarity(
                job_vector, resume_vectors
            ).flatten() * 100

            for (filename, _), category, score in zip(
                readable_resumes, predictions, similarity_scores
            ):
                results.append({
                    "Resume": filename,
                    "Predicted Job Category": category,
                    "Job Description Similarity (%)": round(float(score), 2)
                })

            df = pd.DataFrame(results).sort_values(
                "Job Description Similarity (%)",
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
                "The SVM predicts one of the job categories learned from "
                "the Kaggle dataset. Similarity is a separate text-match "
                "measure. Neither value alone determines hiring suitability; "
                "review resumes manually."
            )
