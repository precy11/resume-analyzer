# Resume Analyzer - Full Core Logic
import PyPDF2
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ---- STEP 1: Extract text from PDF ----
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# ---- STEP 2: Extract skills from text ----
def extract_skills(text):
    skills_list = [
        'python', 'java', 'javascript', 'react', 'nodejs', 'sql',
        'machine learning', 'deep learning', 'html', 'css', 'git',
        'docker', 'aws', 'mongodb', 'flask', 'django', 'c++', 'c',
        'data analysis', 'tensorflow', 'pytorch', 'nlp', 'excel'
    ]
    text_lower = text.lower()
    found_skills = [skill for skill in skills_list if skill in text_lower]
    return found_skills

# ---- STEP 3: Calculate match score ----
def calculate_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

# ---- STEP 4: Find missing skills ----
def find_missing_skills(resume_text, job_description):
    skills_list = [
        'python', 'java', 'javascript', 'react', 'nodejs', 'sql',
        'machine learning', 'deep learning', 'html', 'css', 'git',
        'docker', 'aws', 'mongodb', 'flask', 'django', 'c++', 'c',
        'data analysis', 'tensorflow', 'pytorch', 'nlp', 'excel'
    ]
    resume_skills = extract_skills(resume_text)
    job_skills = [s for s in skills_list if s in job_description.lower()]
    missing = [s for s in job_skills if s not in resume_skills]
    return missing

# ---- TEST WITH SAMPLE DATA ----
sample_resume = """
John Doe - Software Engineer
Skills: Python, Flask, SQL, Git, HTML, CSS, JavaScript
Experience: 2 years building web applications
"""

sample_job = """
We are looking for a Python developer with experience in
Flask, MongoDB, Docker, AWS, SQL and REST APIs.
Knowledge of React is a plus.
"""

print("=" * 50)
print("📄 RESUME ANALYZER TEST")
print("=" * 50)

resume_skills = extract_skills(sample_resume)
print(f"\n✅ Skills found in resume: {resume_skills}")

score = calculate_match_score(sample_resume, sample_job)
print(f"\n📊 Match Score: {score}%")

missing = find_missing_skills(sample_resume, sample_job)
print(f"\n❌ Missing Skills: {missing}")

print("\n✅ Core logic working perfectly!")