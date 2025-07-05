import streamlit as st
import fitz  # PyMuPDF
import docx
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined skill keywords (can be expanded or replaced with NLP-based extraction)
COMMON_SKILLS = [
    "Python", "Java", "C++", "SQL", "JavaScript", "HTML", "CSS", "AWS", "Azure", "Docker", "Kubernetes",
    "Agile", "Scrum", "JIRA", "Project Management", "Leadership", "Communication", "Testing", "Selenium",
    "CI/CD", "Machine Learning", "Data Analysis", "DevOps", "REST", "API", "Git", "Linux", "Cloud"
]

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

# Extract skills from text
def extract_skills(text):
    found_skills = set()
    for skill in COMMON_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_skills.add(skill)
    return found_skills

# Compute semantic similarity
def compute_semantic_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    return float(util.cos_sim(embeddings[0], embeddings[1])[0])

# Streamlit UI
st.title("📄 Generalized Resume Screening Tool")

st.sidebar.header("Upload Files")
job_description_file = st.sidebar.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
resume_files = st.sidebar.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if job_description_file and resume_files:
    st.subheader("📋 Screening Results")

    job_description_text = extract_text(job_description_file)
    jd_skills = extract_skills(job_description_text)

    results = []

    for resume_file in resume_files:
        resume_text = extract_text(resume_file)
        semantic_score = compute_semantic_similarity(resume_text, job_description_text)

        resume_skills = extract_skills(resume_text)
        matched_skills = resume_skills & jd_skills
        missing_skills = jd_skills - resume_skills

        skill_match_score = len(matched_skills) / len(jd_skills) if jd_skills else 0
        final_score = 0.6 * semantic_score + 0.4 * skill_match_score

        results.append({
            "name": resume_file.name,
            "semantic_score": semantic_score,
            "skill_match_score": skill_match_score,
            "final_score": final_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    for res in results:
        st.markdown(f"### {res['name']}")
        st.write(f"**Relevance Score:** {res['final_score']:.2f}")
        st.write(f"**✅ Matched Skills:** {', '.join(sorted(res['matched_skills'])) if res['matched_skills'] else 'None'}")
        st.write(f"**❌ Missing Skills:** {', '.join(sorted(res['missing_skills'])) if res['missing_skills'] else 'None'}")
else:
    st.info("Please upload a job description and at least one resume to begin screening.")

