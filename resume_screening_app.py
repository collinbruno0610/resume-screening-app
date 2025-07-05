import streamlit as st
import fitz  # PyMuPDF
import docx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# General text extractor
def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        return ""

# Compute similarity score
def compute_similarity(resume_text, job_text):
    vectorizer = CountVectorizer().fit_transform([resume_text, job_text])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Streamlit UI
st.title("📄 Resume Screening Tool")

st.sidebar.header("Upload Files")
job_description_file = st.sidebar.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
resume_files = st.sidebar.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if job_description_file and resume_files:
    st.subheader("📋 Screening Results")

    job_description_text = extract_text(job_description_file)
    results = []

    for resume_file in resume_files:
        resume_text = extract_text(resume_file)
        score = compute_similarity(resume_text, job_description_text)
        results.append((resume_file.name, score))

    results.sort(key=lambda x: x[1], reverse=True)

    for name, score in results:
        st.write(f"**{name}** - Relevance Score: {score:.2f}")
else:
    st.info("Please upload a job description and at least one resume to begin screening.")
