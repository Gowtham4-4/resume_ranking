import streamlit as st
import pandas as pd
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle None values
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to highlight keywords in text
def highlight_keywords(text, keywords):
    for word in keywords:
        text = re.sub(f"\\b{word}\\b", f"**{word}**", text, flags=re.IGNORECASE)
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit App UI
st.title("📄 AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# Experience & Skills Filtering
min_experience = st.slider("Minimum Years of Experience", 0, 20, 2)
required_skills = st.text_input("Enter required skills (comma-separated)").split(",")

# File uploader for resumes
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("🔍 Ranking Resumes...")

    resumes = []
    file_names = []

    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.name.endswith(".docx"):
            text = extract_text_from_docx(file)
        else:
            st.warning(f"❌ Unsupported file type: {file.name}")
            continue
        
        resumes.append(text)
        file_names.append(file.name)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Create results DataFrame
    results = pd.DataFrame({"Resume": file_names, "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    # Display results
    st.subheader("🏆 Top Ranked Resumes")
    top_n = min(len(results), 3)  # Show top 3 resumes
    for i, row in results.head(top_n).iterrows():
        st.write(f"### {row['Resume']} (Score: {row['Score']:.2f})")
        highlighted_resume = highlight_keywords(resumes[i], required_skills)
        st.write(highlighted_resume[:1000])  # Show first 1000 characters
        st.markdown("---")

    # CSV Download Option
    csv = results.to_csv(index=False)
    st.download_button(
        label="📥 Download Ranking Results",
        data=io.StringIO(csv).read(),
        file_name="resume_ranking_results.csv",
        mime="text/csv"
    )
