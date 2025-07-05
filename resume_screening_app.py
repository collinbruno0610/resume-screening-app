import streamlit as st
import fitz  # PyMuPDF
import docx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Extract skills from text using simple keyword matching
def extract_skills(text, skill_keywords):
    text_lower = text.lower()
    return [skill for skill in skill_keywords if skill.lower() in text_lower]

# Compute semantic similarity score
def compute_semantic_similarity(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

# Compute skill match score
def compute_skill_match(resume_skills, job_skills):
    if not job_skills:
        return 0
    matched = set(resume_skills).intersection(set(job_skills))
    return len(matched) / len(job_skills)

# Streamlit UI
st.title("📄 Resume Screening Tool with Skill Matching and Visualization")

st.sidebar.header("Upload Files")
job_description_file = st.sidebar.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
resume_files = st.sidebar.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# Define a sample skill list for demonstration
skill_keywords = [
    "Agile", "Scrum", "JIRA", "Leadership", "Project Management", "Stakeholder Communication",
    "Python", "Java", "Selenium", "Test Automation", "CI/CD", "DevOps", "Machine Learning"
]

if job_description_file and resume_files:
    st.subheader("📋 Screening Results")

    job_description_text = extract_text(job_description_file)
    job_skills = extract_skills(job_description_text, skill_keywords)

    results = []
    resume_names = []
    final_scores = []

    for resume_file in resume_files:
        resume_text = extract_text(resume_file)
        resume_skills = extract_skills(resume_text, skill_keywords)

        semantic_score = compute_semantic_similarity(resume_text, job_description_text)
        skill_score = compute_skill_match(resume_skills, job_skills)
        final_score = 0.6 * semantic_score + 0.4 * skill_score

        matched_skills = list(set(resume_skills).intersection(set(job_skills)))
        missing_skills = list(set(job_skills) - set(resume_skills))

        results.append((resume_file.name, final_score, matched_skills, missing_skills))
        resume_names.append(resume_file.name)
        final_scores.append(final_score)

    # Sort results by final score
    results.sort(key=lambda x: x[1], reverse=True)

    for name, score, matched, missing in results:
        st.markdown(f"### {name}")
        st.write(f"**Relevance Score:** {score:.2f}")
        st.write("**✅ Matched Skills:**")
        for skill in matched:
            st.write(f"- {skill}")
        st.write("**❌ Missing Skills:**")
        for skill in missing:
            st.write(f"- {skill}")

    # Bar chart of scores
    st.subheader("📊 Resume Relevance Scores")
    fig, ax = plt.subplots()
    ax.bar(resume_names, final_scores, color='skyblue')
    ax.set_ylabel("Relevance Score")
    ax.set_xlabel("Resume")
    ax.set_title("Resume Relevance Scores")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
else:
    st.info("Please upload a job description and at least one resume to begin screening.")

