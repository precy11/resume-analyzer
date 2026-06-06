# Resume Analyzer - Web UI
import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Helper Functions ----
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_skills(text):
    skills_list = [
        'python', 'java', 'javascript', 'react', 'nodejs', 'sql',
        'machine learning', 'deep learning', 'html', 'css', 'git',
        'docker', 'aws', 'mongodb', 'flask', 'django', 'c++',
        'data analysis', 'tensorflow', 'pytorch', 'nlp', 'excel'
    ]
    text_lower = text.lower()
    return [skill for skill in skills_list if skill in text_lower]

def calculate_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

def find_missing_skills(resume_text, job_description):
    skills_list = [
        'python', 'java', 'javascript', 'react', 'nodejs', 'sql',
        'machine learning', 'deep learning', 'html', 'css', 'git',
        'docker', 'aws', 'mongodb', 'flask', 'django', 'c++',
        'data analysis', 'tensorflow', 'pytorch', 'nlp', 'excel'
    ]
    resume_skills = extract_skills(resume_text)
    job_skills = [s for s in skills_list if s in job_description.lower()]
    return [s for s in job_skills if s not in resume_skills]

# ---- UI Layout ----
st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 Resume Analyzer & Job Matcher")
st.markdown("Upload your resume and paste a job description to see how well you match!")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

with col2:
    st.subheader("💼 Paste Job Description")
    job_description = st.text_area("Job Description", height=200,
                    placeholder="Paste the job description here...")

st.divider()

if st.button("🔍 Analyze Resume", use_container_width=True):
    if uploaded_file and job_description:
        with st.spinner("Analyzing your resume..."):

            resume_text = extract_text_from_pdf(uploaded_file)
            score = calculate_match_score(resume_text, job_description)
            resume_skills = extract_skills(resume_text)
            missing_skills = find_missing_skills(resume_text, job_description)

            # Results
            st.subheader("📊 Results")
            col3, col4, col5 = st.columns(3)

            with col3:
                st.metric("Match Score", f"{score}%")

            with col4:
                st.metric("Skills Found", len(resume_skills))

            with col5:
                st.metric("Missing Skills", len(missing_skills))

            st.divider()

            col6, col7 = st.columns(2)

            with col6:
                st.subheader("✅ Your Skills")
                for skill in resume_skills:
                    st.success(skill)

            with col7:
                st.subheader("❌ Missing Skills")
                if missing_skills:
                    for skill in missing_skills:
                        st.error(skill)
                else:
                    st.balloons()
                    st.success("You have all the required skills!")
    else:
        st.warning("Please upload a resume AND paste a job description!")
        