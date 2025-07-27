import streamlit as st
import pandas as pd
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io
from PIL import Image
# Set Page Config
st.set_page_config(page_title="AI Resume Screener", layout="centered")

# Display Logo
logo = Image.open("1.png")
  # adjust width as needed




# Theme Switcher
theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"])

# Apply CSS Based on Theme
if theme == "Dark":
    background_css = """
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(to bottom right, #000000, #3a3a3a);
            color: white;
        }
        .block-container { max-width: 800px; margin: auto; }
        h1, h2, h3 { color: #4B8BBE; }
        input, textarea { background-color: #222 !important; color: white !important; }
        .stButton>button { background-color: #4B8BBE; color: white; }
        .stButton>button:hover { background-color: #36689a; }
    </style>
    """
else:
    background_css = """
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(to bottom right, #f2f2f2, #cccccc);
            color: black;
        }
        .block-container { max-width: 800px; margin: auto; }
        h1, h2, h3 { color: #1a1a1a; }
        input, textarea { background-color: #ffffff !important; color: black !important; }
        .stButton>button { background-color: #4B8BBE; color: white; }
        .stButton>button:hover { background-color: #36689a; }
    </style>
    """

st.markdown(background_css, unsafe_allow_html=True)

# ---- Logo / Header Image ----
st.image("1.png", width=100)  # You can change this to your own image
st.title("📄 AI Resume Screening & Candidate Ranking System")

# Sidebar Inputs
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    min_experience = st.slider("Minimum Years of Experience", 0, 20, 2)
    required_skills = st.text_input("Required Skills (comma-separated)", "Python, ML, SQL").split(",")

# ----- Functions -----
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def highlight_keywords(text, keywords):
    for word in keywords:
        word = word.strip()
        text = re.sub(f"\\b{re.escape(word)}\\b", f"**{word}**", text, flags=re.IGNORECASE)
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_vector], resume_vectors).flatten()

# Job Description
st.header("📌 Job Description")
job_description = st.text_area("Enter the job description", height=200)

# File Upload
st.header("📁 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF or DOCX resumes", type=["pdf", "docx"], accept_multiple_files=True)

# Main Logic
if uploaded_files and job_description:
    st.header("🔍 Screening and Ranking Resumes...")
    progress = st.progress(0)
    resumes = []
    file_names = []

    for i, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file) if file.name.endswith(".pdf") else extract_text_from_docx(file)
        resumes.append(text)
        file_names.append(file.name)
        progress.progress((i + 1) / len(uploaded_files))

    scores = rank_resumes(job_description, resumes)
    results = pd.DataFrame({"Resume": file_names, "Score": scores}).sort_values(by="Score", ascending=False)

    st.success("✅ Ranking complete!")

    # Show Top Resumes
    st.subheader("🏆 Top Ranked Resumes")
    for i, row in results.head(3).iterrows():
        with st.expander(f"📄 {row['Resume']} (Score: {row['Score']:.2f})"):
            highlighted = highlight_keywords(resumes[i], required_skills)
            st.markdown(highlighted[:1200] + "..." if len(highlighted) > 1200 else highlighted, unsafe_allow_html=True)

    # Chart
    st.subheader("📊 Resume Similarity Scores")
    st.bar_chart(results.set_index("Resume"))

    # Download Button
    st.subheader("📥 Export Results")
    csv = results.to_csv(index=False)
    st.download_button("Download CSV", data=io.StringIO(csv).read(), file_name="resume_scores.csv", mime="text/csv")
else:
    st.info("📌 Please upload resumes and enter a job description to begin.")

# ---- Footer ----
st.markdown("""
---
<div style='text-align: center; font-size: 14px;'>
    Developed by Gowtham 💼 | Contact: <a href="mailto:gowthamperumallapalli@gmail.com">gowthamperumallapalli@gmail.com</a> | <a href="https://linkedin.com/in/yourprofile" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
