import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components
from collections import Counter
import re

st.set_page_config(page_title="ResumeIQ", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")

def extract_text_from_pdf(uploaded_file):
    try:
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t
        return text.strip()
    except Exception:
        return ""

# ── EXPANDED SKILLS LIST (100+) ──────────────────────────────
SKILLS_LIST = [
    # Languages
    'python','java','javascript','typescript','c++','c#','c','ruby','php',
    'swift','kotlin','go','rust','scala','r','matlab','perl','bash','shell',
    # Frontend
    'react','vue','angular','html','css','tailwind','sass','bootstrap',
    'nextjs','gatsby','webpack','jquery','redux',
    # Backend
    'nodejs','flask','django','fastapi','spring','spring boot','laravel',
    'express','rails','asp.net','hibernate','graphql','rest api','soap',
    # Databases
    'sql','postgresql','mysql','mongodb','redis','elasticsearch','sqlite',
    'oracle','firebase','dynamodb','cassandra','neo4j',
    # Cloud & DevOps
    'aws','gcp','azure','docker','kubernetes','terraform','ansible',
    'jenkins','gitlab','github actions','ci/cd','linux','unix','nginx',
    'apache','microservices','serverless',
    # Data & ML
    'machine learning','deep learning','nlp','tensorflow','pytorch',
    'scikit-learn','pandas','numpy','matplotlib','seaborn','keras',
    'data analysis','data science','tableau','power bi','spark','kafka',
    'hadoop','airflow','opencv','computer vision',
    # Mobile
    'android','ios','flutter','react native','xamarin',
    # Tools
    'git','github','jira','figma','postman','selenium','junit',
    'pytest','docker compose','vim','excel','photoshop',
]

ROADMAP = {
    'python':('Python','https://www.learnpython.org/'),
    'java':('Java','https://www.learnjavaonline.org/'),
    'javascript':('JavaScript','https://javascript.info/'),
    'typescript':('TypeScript','https://www.typescriptlang.org/docs/'),
    'react':('React','https://react.dev/learn'),
    'vue':('Vue','https://vuejs.org/tutorial/'),
    'angular':('Angular','https://angular.io/tutorial'),
    'nodejs':('Node.js','https://nodejs.dev/en/learn/'),
    'sql':('SQL','https://sqlzoo.net/'),
    'mongodb':('MongoDB','https://learn.mongodb.com/'),
    'redis':('Redis','https://redis.io/docs/getting-started/'),
    'git':('Git','https://learngitbranching.js.org/'),
    'docker':('Docker','https://docs.docker.com/get-started/'),
    'kubernetes':('Kubernetes','https://kubernetes.io/docs/tutorials/'),
    'aws':('AWS','https://aws.amazon.com/training/'),
    'gcp':('GCP','https://cloud.google.com/training'),
    'azure':('Azure','https://learn.microsoft.com/en-us/azure/'),
    'flask':('Flask','https://flask.palletsprojects.com/'),
    'django':('Django','https://docs.djangoproject.com/en/5.0/intro/'),
    'fastapi':('FastAPI','https://fastapi.tiangolo.com/tutorial/'),
    'html':('HTML','https://www.freecodecamp.org/learn/2022/responsive-web-design/'),
    'css':('CSS','https://web.dev/learn/css/'),
    'tailwind':('Tailwind','https://tailwindcss.com/docs/'),
    'machine learning':('ML','https://www.coursera.org/learn/machine-learning'),
    'deep learning':('Deep Learning','https://www.deeplearning.ai/'),
    'nlp':('NLP','https://huggingface.co/learn/nlp-course/'),
    'tensorflow':('TensorFlow','https://www.tensorflow.org/tutorials'),
    'pytorch':('PyTorch','https://pytorch.org/tutorials/'),
    'scikit-learn':('Scikit-learn','https://scikit-learn.org/stable/tutorial/'),
    'pandas':('Pandas','https://pandas.pydata.org/docs/getting_started/'),
    'numpy':('NumPy','https://numpy.org/learn/'),
    'linux':('Linux','https://linuxjourney.com/'),
    'graphql':('GraphQL','https://graphql.org/learn/'),
    'rest api':('REST API','https://restfulapi.net/'),
    'spring boot':('Spring Boot','https://spring.io/guides'),
    'kotlin':('Kotlin','https://kotlinlang.org/docs/getting-started.html'),
    'flutter':('Flutter','https://flutter.dev/docs/get-started'),
    'selenium':('Selenium','https://www.selenium.dev/documentation/'),
    'postman':('Postman','https://learning.postman.com/'),
    'figma':('Figma','https://www.figma.com/resources/learn-design/'),
    'tableau':('Tableau','https://www.tableau.com/learn/training'),
    'power bi':('Power BI','https://learn.microsoft.com/en-us/power-bi/'),
    'firebase':('Firebase','https://firebase.google.com/docs'),
    'spark':('Apache Spark','https://spark.apache.org/docs/latest/'),
    'kafka':('Kafka','https://kafka.apache.org/documentation/'),
}

def extract_skills(text):
    t = text.lower()
    return [s for s in SKILLS_LIST if s in t]

def calculate_match(resume_text, jd):
    v = TfidfVectorizer(stop_words='english')
    m = v.fit_transform([resume_text, jd])
    return round(cosine_similarity(m[0], m[1])[0][0] * 100, 1)

def find_missing(resume_text, jd):
    have = set(extract_skills(resume_text))
    needed = [s for s in SKILLS_LIST if s in jd.lower()]
    return [s for s in needed if s not in have]

def score_label(score):
    if score >= 65: return "Strong Match 🎯","Your profile aligns well. Apply with confidence."
    if score >= 40: return "Moderate Match ⚡","Solid base — bridge a few gaps before applying."
    return "Weak Match 🔍","Significant skill gaps detected. Keep building."

def score_color(score):
    if score >= 65: return "#00ffe7"
    if score >= 40: return "#fbbf24"
    return "#f87171"

def ats_score(resume_text, filename):
    score = 60
    tips = []
    if filename.endswith('.pdf'): score += 10
    else: tips.append("Use PDF format for better ATS compatibility")
    text_lower = resume_text.lower()
    kw = ['experience','education','skills','project','summary','objective']
    score += sum(1 for k in kw if k in text_lower) * 3
    if len(resume_text) > 2000: score += 5
    else: tips.append("Add more detail — ATS prefers longer keyword-rich resumes")
    if any(c in resume_text for c in ['|','•','▪','→']): score += 5
    if len(resume_text) > 5000:
        score -= 10
        tips.append("Resume might be too long — keep it under 2 pages")
    score = min(score, 100)
    if score >= 75: tips.append("Good ATS compatibility — keep using standard section headers")
    elif score >= 55:
        tips.append("Add more keywords from the job description")
        tips.append("Use standard headers: Experience, Education, Skills")
    else:
        tips.append("Add standard sections: Summary, Experience, Education, Skills")
        tips.append("Include more relevant keywords from the job description")
    return min(score,100), tips

def improvement_suggestions(have, missing, score):
    tips = []
    if score < 40: tips.append("Overall match is low — tailor your resume specifically for this role")
    if len(missing) > 5: tips.append(f"Focus on top missing skills first: {', '.join(missing[:3])}")
    if 'git' in missing: tips.append("Add Git — it is expected in almost every tech role")
    if 'docker' in missing: tips.append("Learn Docker basics — containerization is now a core dev skill")
    if any(s in missing for s in ['aws','gcp','azure']): tips.append("Cloud skills (AWS/GCP/Azure) are highly valued — start with free tier")
    if len(have) < 5: tips.append("List more technical skills — even basics like HTML, Git count")
    if score >= 65:
        tips.append("Strong match! Highlight your most relevant projects at the top")
        tips.append("Quantify your experience — use numbers like '30% faster', '10k users'")
    if not tips: tips.append("Good match! Customize your resume summary to mirror the job description")
    return tips[:5]

# FIX 5: Extract top keywords from JD
def extract_top_keywords(jd_text):
    stop = {'the','and','or','for','with','you','are','will','have','that',
            'this','from','our','we','to','a','an','in','of','is','be','as',
            'on','at','by','it','its','your','their','all','also','not','but',
            'can','was','has','more','than','into','about','they','who','which'}
    words = re.findall(r'\b[a-zA-Z][a-zA-Z+#.]{2,}\b', jd_text.lower())
    filtered = [w for w in words if w not in stop and len(w) > 2]
    return [w for w,_ in Counter(filtered).most_common(12)]

# FIX 4: Resume stats
def resume_stats(text):
    words = len(text.split())
    pages = round(words / 350, 1)
    chars = len(text)
    return words, pages, chars

# ── Global Streamlit styles ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important;}
.stApp{background:#060608!important;}

/* Theme toggle button */
.theme-toggle-wrap {
    position: fixed;
    top: 16px;
    right: 20px;
    z-index: 9999;
    display: flex;
    gap: 4px;
    background: var(--bg2, #0d1a1a);
    border: 1px solid var(--border, #0a3a3a);
    border-radius: 12px;
    padding: 4px;
}
.theme-btn {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.06em;
    padding: 6px 12px;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: var(--text3, #1a5050);
    cursor: pointer;
    transition: all 0.2s;
}
.theme-btn:hover { color: var(--text, #00ffe7); }
.theme-btn.active {
    background: rgba(0,255,200,0.12);
    color: var(--text, #00ffe7);
    border: 1px solid rgba(0,255,200,0.2);
}

/* Light theme Streamlit overrides */
[data-theme="light"] .stApp { background: #eaf6f6 !important; }
[data-theme="light"] .main .block-container { background: #eaf6f6 !important; }
[data-theme="light"] section[data-testid="stMain"] { background: #eaf6f6 !important; }
[data-theme="light"] .stWarning { background: #fff8e1 !important; border: 1px solid #f0c040 !important; border-radius: 12px !important; }
[data-theme="light"] .stWarning p { color: #7a5000 !important; font-family: 'Space Grotesk',sans-serif !important; font-size: 13px !important; }
[data-theme="light"] .stWarning svg { fill: #f0a000 !important; }
[data-theme="light"] .stFileUploader > div { background: #d0ecec !important; border-color: #80cccc !important; }
[data-theme="light"] .stFileUploader > div > div { background: #d0ecec !important; }
[data-theme="light"] .stFileUploader section { background: #d0ecec !important; border-color: #80cccc !important; }
[data-theme="light"] [data-testid="stFileUploaderDropzone"] { background: #d0ecec !important; border-color: #80cccc !important; }
[data-theme="light"] [data-testid="stFileUploaderDropzone"] > div { background: #d0ecec !important; }
[data-theme="light"] .stFileUploader * { color: #007a7a !important; }
[data-theme="light"] .stFileUploader button { background: #007a7a !important; color: #eaf6f6 !important; border-color: #007a7a !important; }
[data-theme="light"] .stFileUploader label { color: #007a7a !important; }
[data-theme="light"] .stTextArea textarea { background: #e0f5f5 !important; border-color: #a0d8d8 !important; color: #007a7a !important; }
[data-theme="light"] .score-card { background: #e0f5f5 !important; border-color: #a0d8d8 !important; }
[data-theme="light"] .stat { background: #e0f5f5 !important; border-color: #a0d8d8 !important; }
[data-theme="light"] .feat-card { background: #e0f5f5 !important; border-color: #a0d8d8 !important; }
[data-theme="light"] .score-verdict { color: #007a7a !important; }
[data-theme="light"] .score-desc { color: #005555 !important; }
[data-theme="light"] .score-explain { color: #aacccc !important; }
[data-theme="light"] .ring-num { color: #007a7a !important; }
[data-theme="light"] .stat-l { color: #aaaaaa !important; }
[data-theme="light"] .tip-text { color: #005555 !important; }
[data-theme="light"] .road-skill { color: #cc3333 !important; }
[data-theme="light"] .breakdown-track { background: #c0e8e8 !important; }
[data-theme="light"] .breakdown-label { color: #007a7a !important; }
[data-theme="light"] .feat-title { color: #007a7a !important; }
[data-theme="light"] .sec-label { color: #007a7a !important; }
[data-theme="light"] .ok-file { color: #007a7a !important; }
[data-theme="light"] .chip.have { background: rgba(0,150,150,0.08) !important; border-color: rgba(0,150,150,0.3) !important; color: #007a7a !important; }
[data-theme="light"] .kw-chip { background: rgba(100,80,200,0.08) !important; border-color: rgba(100,80,200,0.2) !important; color: #5040a0 !important; }
[data-theme="light"] .resume-stat { color: #007a7a !important; background: rgba(0,150,150,0.06) !important; border-color: rgba(0,150,150,0.15) !important; }
[data-theme="light"] .ats-bar { background: #c0e8e8 !important; }

section[data-testid="stSidebar"]{display:none;}
#MainMenu,footer,header{visibility:hidden;}
.main .block-container{padding:1rem 3rem 4rem!important;max-width:960px!important;margin:0 auto!important;}
.stFileUploader>div{background:#0d1a1a!important;border:1.5px dashed #0a3a3a!important;border-radius:14px!important;}
.stFileUploader>div:hover{border-color:#00ffe7!important;}
.stFileUploader label{color:#00ffe7!important;font-family:'Space Mono',monospace!important;font-size:10px!important;letter-spacing:0.12em!important;}
.stFileUploader button{background:#00ffe7!important;color:#060608!important;border:none!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:10px!important;font-weight:700!important;}
.stFileUploader p{color:#1a6060!important;}
[data-theme="light"] .stFileUploader>div{background:#d0ecec!important;border-color:#80cccc!important;}
[data-theme="light"] .stFileUploader>div:hover{border-color:#007a7a!important;}
[data-theme="light"] .stFileUploader label{color:#007a7a!important;}
[data-theme="light"] .stFileUploader p{color:#007a7a!important;}
[data-theme="light"] .stFileUploader small{color:#1a8080!important;}
[data-theme="light"] .stFileUploader button{background:#007a7a!important;color:#eaf6f6!important;}
[data-theme="light"] [data-testid="stFileUploaderDropzone"]{background:#d0ecec!important;border-color:#80cccc!important;}
[data-theme="light"] [data-testid="stFileUploaderDropzone"] *{color:#007a7a!important;}
[data-theme="light"] [data-testid="stFileUploaderDropzone"] button{background:#007a7a!important;color:#eaf6f6!important;}
[data-theme="light"] .stFileUploader section{background:#d0ecec!important;}
.stTextArea textarea{background:#0d1a1a!important;border:1.5px solid #0a3a3a!important;border-radius:12px!important;color:#00ffe7!important;font-family:'Space Grotesk',sans-serif!important;font-size:13px!important;line-height:1.65!important;padding:12px 14px!important;resize:none!important;}
.stTextArea textarea:focus{border-color:#00ffe7!important;box-shadow:0 0 0 3px rgba(0,255,231,0.08)!important;}
.stTextArea textarea::placeholder{color:#0a2a2a!important;}
.stTextArea label{color:#00ffe7!important;font-family:'Space Mono',monospace!important;font-size:10px!important;letter-spacing:0.12em!important;}
.stButton>button{width:100%!important;background:linear-gradient(135deg,#00ffe7,#00bcd4)!important;color:#060608!important;border:none!important;border-radius:14px!important;font-family:'Space Grotesk',sans-serif!important;font-size:15px!important;font-weight:700!important;padding:16px!important;transition:all 0.2s!important;}
.stButton>button:hover{opacity:0.88!important;transform:translateY(-2px)!important;box-shadow:0 10px 30px rgba(0,255,231,0.2)!important;}
.sec-label{font-family:'Space Mono',monospace;font-size:9px;color:#00ffe7;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:8px;opacity:0.5;}
.ok-file{font-family:'Space Mono',monospace;font-size:11px;color:#00ffe7;margin-top:6px;}
.res-wrap{margin-top:28px;animation:fu 0.4s ease;}
@keyframes fu{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.score-card{background:#0d1a1a;border:1px solid #0a3a3a;border-radius:20px;padding:28px;margin-bottom:14px;display:flex;align-items:center;gap:24px;}
.ring{position:relative;width:96px;height:96px;flex-shrink:0;}
.ring svg{transform:rotate(-90deg);}
.ring-num{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:#00ffe7;}
.score-right{flex:1;}
.score-verdict{font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:#00ffe7;margin-bottom:6px;letter-spacing:-0.5px;}
.score-desc{font-family:'Space Grotesk',sans-serif;font-size:13px;color:#1a6060;line-height:1.6;}
.score-explain{font-family:'Space Mono',monospace;font-size:9px;color:#0a3a3a;margin-top:8px;line-height:1.5;}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:14px;}
.chip{font-family:'Space Mono',monospace;padding:4px 13px;border-radius:100px;font-size:10px;}
.have{background:rgba(0,255,231,0.08);border:1px solid rgba(0,255,231,0.25);color:#00ffe7;}
.miss{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.25);color:#f87171;}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px;}
.stat{background:#0d1a1a;border:1px solid #0a3a3a;border-radius:14px;padding:20px;text-align:center;}
.stat-n{font-family:'Space Grotesk',sans-serif;font-size:32px;font-weight:700;line-height:1;letter-spacing:-1px;}
.stat-l{font-family:'Space Mono',monospace;font-size:9px;color:#1a5050;letter-spacing:0.14em;text-transform:uppercase;margin-top:6px;}
.feat-card{background:#0d1a1a;border:1px solid #0a3a3a;border-radius:20px;padding:24px;margin-bottom:14px;}
.feat-title{font-family:'Space Mono',monospace;font-size:9px;color:#00ffe7;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:16px;opacity:0.6;}
.ats-wrap{display:flex;align-items:center;gap:16px;margin-bottom:14px;}
.ats-num{font-family:'Space Grotesk',sans-serif;font-size:40px;font-weight:700;letter-spacing:-2px;flex-shrink:0;}
.ats-right{flex:1;}
.ats-label{font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;margin-bottom:6px;}
.ats-bar{height:6px;background:#0a2a2a;border-radius:100px;overflow:hidden;}
.ats-fill{height:100%;border-radius:100px;}
.tip-item{display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid #0a2a2a;}
.tip-item:last-child{border-bottom:none;}
.tip-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:5px;}
.tip-text{font-family:'Space Grotesk',sans-serif;font-size:13px;color:#1a8080;line-height:1.5;}
.road-item{display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid #0a2a2a;}
.road-item:last-child{border-bottom:none;}
.road-skill{font-family:'Space Mono',monospace;font-size:11px;color:#f87171;}
.road-link{font-family:'Space Mono',monospace;font-size:10px;color:#00ffe7;text-decoration:none;padding:3px 10px;border:1px solid rgba(0,255,231,0.2);border-radius:6px;transition:all 0.2s;}
.road-link:hover{background:rgba(0,255,231,0.1);}
.breakdown-row{margin-bottom:12px;}
.breakdown-label{display:flex;justify-content:space-between;font-family:'Space Mono',monospace;font-size:10px;color:#1a6060;margin-bottom:5px;}
.breakdown-track{height:6px;background:#0a2a2a;border-radius:100px;overflow:hidden;}
.breakdown-fill{height:100%;border-radius:100px;transition:width 1s cubic-bezier(0.4,0,0.2,1);}
.kw-chip{font-family:'Space Mono',monospace;font-size:10px;padding:4px 12px;border-radius:100px;background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);color:#a78bfa;margin:3px;}
.resume-stat{display:inline-block;font-family:'Space Mono',monospace;font-size:10px;color:#00ffe7;background:rgba(0,255,200,0.06);border:1px solid rgba(0,255,200,0.15);padding:5px 14px;border-radius:8px;margin-right:8px;margin-bottom:6px;}
.error-box{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.25);border-radius:14px;padding:18px 20px;color:#f87171;font-family:'Space Grotesk',sans-serif;font-size:13px;line-height:1.6;margin-top:14px;}
.stWarning{background:#fff3cd!important;border:1.5px solid #e0a800!important;border-radius:12px!important;}
.stWarning *{color:#7a4800!important;font-family:'Space Grotesk',sans-serif!important;font-size:14px!important;font-weight:600!important;opacity:1!important;}
div[data-testid="stAlert"]{background:#fff3cd!important;border:1.5px solid #e0a800!important;border-radius:12px!important;}
div[data-testid="stAlert"] *{color:#7a4800!important;font-size:14px!important;font-weight:600!important;opacity:1!important;}
div[data-testid="stAlert"][kind="warning"]{background:#fff3cd!important;}
div[data-testid="stAlert"][kind="warning"] p{color:#7a4800!important;opacity:1!important;}
div[role="alert"]{background:#fff3cd!important;border:1.5px solid #e0a800!important;border-radius:12px!important;}
div[role="alert"] *{color:#7a4800!important;opacity:1!important;font-size:14px!important;font-weight:600!important;}
#global-overlay{position:fixed;inset:0;width:100vw;height:100vh;background:rgba(0,0,0,0.78);backdrop-filter:blur(8px);z-index:99999;display:none;align-items:center;justify-content:center;}
#global-overlay.show{display:flex;}
#global-modal-box{background:#ffffff;border:1px solid #80cccc;border-radius:20px;padding:36px;max-width:520px;width:90%;max-height:82vh;overflow-y:auto;position:relative;animation:gmIn 0.25s ease;font-family:'Space Grotesk',sans-serif;}
[data-theme="dark"] #global-modal-box{background:#0a1414;border-color:#0a3a3a;}
@keyframes gmIn{from{opacity:0;transform:scale(0.95)}to{opacity:1;transform:scale(1)}}
.gm-close{position:absolute;top:16px;right:16px;background:transparent;border:none;font-size:20px;cursor:pointer;color:#777;line-height:1;transition:color 0.2s;}
.gm-close:hover{color:#003333;}
[data-theme="dark"] .gm-close:hover{color:#00ffe7;}
.gm-title{font-size:24px;font-weight:800;color:#003333;margin-bottom:6px;letter-spacing:-0.5px;}
[data-theme="dark"] .gm-title{color:#00ffe7;}
.gm-sub{font-family:'Space Mono',monospace;font-size:9px;color:#888;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:24px;}
.gm-step{display:flex;gap:14px;margin-bottom:18px;align-items:flex-start;}
.gm-num{width:30px;height:30px;border-radius:8px;background:rgba(0,100,100,0.1);border:1px solid rgba(0,100,100,0.25);color:#005555;font-family:'Space Mono',monospace;font-size:11px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-weight:700;}
[data-theme="dark"] .gm-num{background:rgba(0,255,200,0.08);border-color:rgba(0,255,200,0.2);color:#00ffe7;}
.gm-text{font-size:13px;color:#111;line-height:1.65;}
[data-theme="dark"] .gm-text{color:#1a8080;}
.gm-text strong{color:#006666;}
[data-theme="dark"] .gm-text strong{color:#00ffe7;}
.gm-card{background:#f0fafa;border:1px solid #80cccc;border-radius:12px;padding:16px;margin-bottom:10px;}
[data-theme="dark"] .gm-card{background:#060e0e;border-color:#0a2a2a;}
.gm-role{font-size:15px;font-weight:700;color:#003333;margin-bottom:4px;}
[data-theme="dark"] .gm-role{color:#00ffe7;}
.gm-score{font-family:'Space Mono',monospace;font-size:11px;color:#336666;margin-bottom:8px;}
[data-theme="dark"] .gm-score{color:#1a6060;}
.gm-chips{display:flex;gap:5px;flex-wrap:wrap;}
.gm-chip{font-family:'Space Mono',monospace;font-size:9px;padding:3px 10px;border-radius:100px;}
.gm-have{background:rgba(0,100,100,0.1);border:1px solid rgba(0,100,100,0.25);color:#005555;}
[data-theme="dark"] .gm-have{background:rgba(0,255,200,0.08);border-color:rgba(0,255,200,0.2);color:#00ffe7;}
.gm-miss{background:rgba(160,0,0,0.07);border:1px solid rgba(160,0,0,0.2);color:#aa0000;}
[data-theme="dark"] .gm-miss{background:rgba(248,113,113,0.08);border-color:rgba(248,113,113,0.2);color:#f87171;}
.gm-body{font-size:13px;color:#111;line-height:1.7;margin-bottom:14px;}
[data-theme="dark"] .gm-body{color:#1a8080;}
.gm-tag{display:inline-block;background:rgba(0,100,100,0.1);border:1px solid rgba(0,100,100,0.2);color:#005555;font-family:'Space Mono',monospace;font-size:9px;padding:3px 10px;border-radius:100px;margin:3px 3px 3px 0;}
[data-theme="dark"] .gm-tag{background:rgba(0,255,200,0.08);border-color:rgba(0,255,200,0.2);color:#00ffe7;}
.gm-section{font-size:14px;font-weight:700;color:#006666;margin:16px 0 8px;}
[data-theme="dark"] .gm-section{color:#00ffe7;}
</style>

<div id="global-overlay" onclick="if(event.target===this)closeGlobalModal()">
  <div id="global-modal-box">
    <button class="gm-close" onclick="closeGlobalModal()">✕</button>
    <div id="gm-content"></div>
  </div>
</div>

<script>
function openGlobalModal(type){
  const contents = {
    how: `<div class="gm-title">How it works</div><div class="gm-sub">4 simple steps</div><div class="gm-step"><div class="gm-num">01</div><div class="gm-text"><strong>Upload your resume</strong> — PDF is parsed and all skills, keywords and content are extracted instantly.</div></div><div class="gm-step"><div class="gm-num">02</div><div class="gm-text"><strong>Paste a job description</strong> — Copy any job posting from LinkedIn, Naukri, or any job board.</div></div><div class="gm-step"><div class="gm-num">03</div><div class="gm-text"><strong>Get your analysis</strong> — AI calculates match score, checks ATS compatibility, finds skill gaps and gives a learning roadmap.</div></div><div class="gm-step"><div class="gm-num">04</div><div class="gm-text"><strong>Improve and reapply</strong> — Use suggestions and roadmap to upskill until score is above 65%.</div></div>`,
    examples: `<div class="gm-title">Example Results</div><div class="gm-sub">See what the analysis looks like</div><div class="gm-card"><div class="gm-role">Python Developer @ Startup</div><div class="gm-score">Match Score: 72% · Strong Match 🎯</div><div class="gm-chips"><span class="gm-chip gm-have">✓ python</span><span class="gm-chip gm-have">✓ flask</span><span class="gm-chip gm-have">✓ sql</span><span class="gm-chip gm-have">✓ git</span><span class="gm-chip gm-miss">✗ docker</span><span class="gm-chip gm-miss">✗ aws</span></div></div><div class="gm-card"><div class="gm-role">ML Engineer @ Product Company</div><div class="gm-score">Match Score: 45% · Moderate Match ⚡</div><div class="gm-chips"><span class="gm-chip gm-have">✓ python</span><span class="gm-chip gm-have">✓ pandas</span><span class="gm-chip gm-miss">✗ tensorflow</span><span class="gm-chip gm-miss">✗ pytorch</span><span class="gm-chip gm-miss">✗ kubernetes</span></div></div><div class="gm-card"><div class="gm-role">Frontend Developer @ MNC</div><div class="gm-score">Match Score: 85% · Strong Match 🎯</div><div class="gm-chips"><span class="gm-chip gm-have">✓ react</span><span class="gm-chip gm-have">✓ javascript</span><span class="gm-chip gm-have">✓ typescript</span><span class="gm-chip gm-have">✓ css</span><span class="gm-chip gm-miss">✗ graphql</span></div></div>`,
    about: `<div class="gm-title">About ResumeIQ</div><div class="gm-sub">AI-powered career intelligence</div><div class="gm-body">ResumeIQ is a free AI-powered tool that helps students and job seekers understand exactly how well their resume matches any job description — before they apply.</div><div class="gm-body">Built with Python, NLP and Streamlit. Uses TF-IDF cosine similarity for matching and rule-based ATS scoring.</div><div class="gm-section">Tech Stack</div><div><span class="gm-tag">Python</span><span class="gm-tag">Streamlit</span><span class="gm-tag">Scikit-learn</span><span class="gm-tag">PyPDF2</span><span class="gm-tag">TF-IDF</span><span class="gm-tag">Cosine Similarity</span></div><div class="gm-section">Features</div><div><span class="gm-tag">Match Score</span><span class="gm-tag">ATS Checker</span><span class="gm-tag">Skill Gap Analysis</span><span class="gm-tag">AI Suggestions</span><span class="gm-tag">Learning Roadmap</span><span class="gm-tag">Score Breakdown</span><span class="gm-tag">Top Keywords</span><span class="gm-tag">Resume Stats</span></div>`
  };
  document.getElementById('gm-content').innerHTML = contents[type] || '';
  document.getElementById('global-overlay').classList.add('show');
}
function closeGlobalModal(){
  document.getElementById('global-overlay').classList.remove('show');
}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeGlobalModal();});
// Listen for postMessage from iframes
window.addEventListener('message', function(e){
  if(e.data && e.data.type === 'openModal'){
    openGlobalModal(e.data.modal);
  }
});
</script>
""", unsafe_allow_html=True)

# ── Global Modal System (uses components.html so JS runs) ──────
components.html('''
<!DOCTYPE html>
<html>
<head>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:transparent;overflow:hidden;}
</style>
</head>
<body>
<script>
(function(){
  const p = window.parent.document;
  // Inject modal overlay into parent document body
  if(!p.getElementById("gm-overlay")){
    const style = p.createElement("style");
    style.textContent = `
      #gm-overlay{position:fixed;inset:0;width:100%;height:100%;background:rgba(0,0,0,0.78);backdrop-filter:blur(8px);z-index:999999;display:none;align-items:center;justify-content:center;}
      #gm-overlay.open{display:flex;}
      #gm-box{background:#fff;border:1px solid #80cccc;border-radius:20px;padding:36px;max-width:520px;width:90%;max-height:82vh;overflow-y:auto;position:relative;animation:gmIn .25s ease;font-family:"Space Grotesk",sans-serif;}
      @keyframes gmIn{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
      [data-theme="dark"] #gm-box{background:#0a1414;border-color:#0a3a3a;}
      #gm-close{position:absolute;top:16px;right:16px;background:transparent;border:none;font-size:22px;cursor:pointer;color:#777;line-height:1;}
      #gm-close:hover{color:#003333;}
      [data-theme="dark"] #gm-close:hover{color:#00ffe7;}
      .gm-title{font-size:24px;font-weight:800;color:#003333;margin-bottom:6px;letter-spacing:-.5px;}
      [data-theme="dark"] .gm-title{color:#00ffe7;}
      .gm-sub{font-family:"Space Mono",monospace;font-size:9px;color:#888;letter-spacing:.15em;text-transform:uppercase;margin-bottom:24px;}
      .gm-step{display:flex;gap:14px;margin-bottom:18px;align-items:flex-start;}
      .gm-num{width:30px;height:30px;border-radius:8px;background:rgba(0,100,100,.1);border:1px solid rgba(0,100,100,.25);color:#005555;font-family:"Space Mono",monospace;font-size:11px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-weight:700;}
      [data-theme="dark"] .gm-num{background:rgba(0,255,200,.08);border-color:rgba(0,255,200,.2);color:#00ffe7;}
      .gm-text{font-size:13px;color:#111;line-height:1.65;}
      [data-theme="dark"] .gm-text{color:#1a8080;}
      .gm-text strong{color:#006666;}
      [data-theme="dark"] .gm-text strong{color:#00ffe7;}
      .gm-card{background:#f0fafa;border:1px solid #80cccc;border-radius:12px;padding:16px;margin-bottom:10px;}
      [data-theme="dark"] .gm-card{background:#060e0e;border-color:#0a2a2a;}
      .gm-role{font-size:15px;font-weight:700;color:#003333;margin-bottom:4px;}
      [data-theme="dark"] .gm-role{color:#00ffe7;}
      .gm-score{font-family:"Space Mono",monospace;font-size:11px;color:#336666;margin-bottom:8px;}
      [data-theme="dark"] .gm-score{color:#1a6060;}
      .gm-chips{display:flex;gap:5px;flex-wrap:wrap;}
      .gm-chip{font-family:"Space Mono",monospace;font-size:9px;padding:3px 10px;border-radius:100px;}
      .gm-have{background:rgba(0,100,100,.1);border:1px solid rgba(0,100,100,.25);color:#005555;}
      [data-theme="dark"] .gm-have{background:rgba(0,255,200,.08);border-color:rgba(0,255,200,.2);color:#00ffe7;}
      .gm-miss{background:rgba(160,0,0,.07);border:1px solid rgba(160,0,0,.2);color:#aa0000;}
      [data-theme="dark"] .gm-miss{background:rgba(248,113,113,.08);border-color:rgba(248,113,113,.2);color:#f87171;}
      .gm-body{font-size:13px;color:#111;line-height:1.7;margin-bottom:14px;}
      [data-theme="dark"] .gm-body{color:#1a8080;}
      .gm-tag{display:inline-block;background:rgba(0,100,100,.1);border:1px solid rgba(0,100,100,.2);color:#005555;font-family:"Space Mono",monospace;font-size:9px;padding:3px 10px;border-radius:100px;margin:3px 3px 3px 0;}
      [data-theme="dark"] .gm-tag{background:rgba(0,255,200,.08);border-color:rgba(0,255,200,.2);color:#00ffe7;}
      .gm-section{font-size:14px;font-weight:700;color:#006666;margin:16px 0 8px;}
      [data-theme="dark"] .gm-section{color:#00ffe7;}
    `;
    p.head.appendChild(style);
    const overlay = p.createElement("div");
    overlay.id = "gm-overlay";
    overlay.innerHTML = `<div id="gm-box"><button id="gm-close" onclick="document.getElementById('gm-overlay').classList.remove('open')">✕</button><div id="gm-inner"></div></div>`;
    overlay.addEventListener("click", function(e){ if(e.target===overlay) overlay.classList.remove("open"); });
    p.body.appendChild(overlay);
  }

  const contents = {
    how: `<div class="gm-title">How it works</div><div class="gm-sub">4 simple steps</div><div class="gm-step"><div class="gm-num">01</div><div class="gm-text"><strong>Upload your resume</strong> — PDF is parsed and all skills, keywords and content are extracted instantly.</div></div><div class="gm-step"><div class="gm-num">02</div><div class="gm-text"><strong>Paste a job description</strong> — Copy any job posting from LinkedIn, Naukri, or any job board.</div></div><div class="gm-step"><div class="gm-num">03</div><div class="gm-text"><strong>Get your analysis</strong> — AI calculates match score, checks ATS, finds skill gaps and gives a learning roadmap.</div></div><div class="gm-step"><div class="gm-num">04</div><div class="gm-text"><strong>Improve and reapply</strong> — Use suggestions to upskill until score is above 65%.</div></div>`,
    examples: `<div class="gm-title">Example Results</div><div class="gm-sub">See what the analysis looks like</div><div class="gm-card"><div class="gm-role">Python Developer @ Startup</div><div class="gm-score">Match Score: 72% · Strong Match 🎯</div><div class="gm-chips"><span class="gm-chip gm-have">✓ python</span><span class="gm-chip gm-have">✓ flask</span><span class="gm-chip gm-have">✓ sql</span><span class="gm-chip gm-have">✓ git</span><span class="gm-chip gm-miss">✗ docker</span><span class="gm-chip gm-miss">✗ aws</span></div></div><div class="gm-card"><div class="gm-role">ML Engineer @ Product Company</div><div class="gm-score">Match Score: 45% · Moderate Match ⚡</div><div class="gm-chips"><span class="gm-chip gm-have">✓ python</span><span class="gm-chip gm-have">✓ pandas</span><span class="gm-chip gm-miss">✗ tensorflow</span><span class="gm-chip gm-miss">✗ pytorch</span><span class="gm-chip gm-miss">✗ kubernetes</span></div></div><div class="gm-card"><div class="gm-role">Frontend Developer @ MNC</div><div class="gm-score">Match Score: 85% · Strong Match 🎯</div><div class="gm-chips"><span class="gm-chip gm-have">✓ react</span><span class="gm-chip gm-have">✓ javascript</span><span class="gm-chip gm-have">✓ typescript</span><span class="gm-chip gm-have">✓ css</span><span class="gm-chip gm-miss">✗ graphql</span></div></div>`,
    about: `<div class="gm-title">About ResumeIQ</div><div class="gm-sub">AI-powered career intelligence</div><div class="gm-body">ResumeIQ is a free AI-powered tool that helps students and job seekers understand exactly how well their resume matches any job description — before they apply.</div><div class="gm-body">Built with Python, NLP and Streamlit. Uses TF-IDF cosine similarity for matching and rule-based ATS scoring.</div><div class="gm-section">Tech Stack</div><div><span class="gm-tag">Python</span><span class="gm-tag">Streamlit</span><span class="gm-tag">Scikit-learn</span><span class="gm-tag">PyPDF2</span><span class="gm-tag">TF-IDF</span></div><div class="gm-section">Features</div><div><span class="gm-tag">Match Score</span><span class="gm-tag">ATS Checker</span><span class="gm-tag">Skill Gap Analysis</span><span class="gm-tag">AI Suggestions</span><span class="gm-tag">Learning Roadmap</span><span class="gm-tag">Score Breakdown</span><span class="gm-tag">Top Keywords</span></div>`
  };

  function openModal(type){
    const inner = p.getElementById("gm-inner");
    if(inner) inner.innerHTML = contents[type] || "";
    const ov = p.getElementById("gm-overlay");
    if(ov) ov.classList.add("open");
  }

  // Listen for postMessage from hero iframe
  window.parent.addEventListener("message", function(e){
    if(e.data && e.data.type === "openModal") openModal(e.data.modal);
  });

  // ESC to close
  p.addEventListener("keydown", function(e){
    if(e.key==="Escape") { const ov=p.getElementById("gm-overlay"); if(ov) ov.classList.remove("open"); }
  });
})();
</script>
</body>
</html>
''', height=0, scrolling=False)




# ── Hero Component ───────────────────────────────────────────
components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#060608;font-family:'Space Grotesk',sans-serif;overflow:hidden;cursor:default;transition:background 0.3s;}
body[data-theme="light"]{background:#e8f8f8 !important;}
body[data-theme="light"] .bg{background-image:radial-gradient(circle,#b0d8d8 1px,transparent 1px) !important;opacity:0.3;}
body[data-theme="light"] #sg{background:radial-gradient(circle,rgba(0,150,150,0.1) 0%,transparent 70%) !important;}
body[data-theme="light"] .logo .lr{color:#007a7a !important;}
body[data-theme="light"] .logo .li{color:#009999 !important;}
body[data-theme="light"] .nav-links a{color:#aacccc !important;}
body[data-theme="light"] .nav-links a:hover{color:#007a7a !important;}
body[data-theme="light"] .badge{background:rgba(0,150,150,0.1) !important;border-color:rgba(0,150,150,0.2) !important;color:#007a7a !important;}
body[data-theme="light"] .dot{background:#007a7a !important;box-shadow:0 0 8px #007a7a !important;}
body[data-theme="light"] .w1{color:#007a7a !important;}
body[data-theme="light"] .w2{color:#009999 !important;}
body[data-theme="light"] .sub{color:#aacccc !important;}
body[data-theme="light"] .act-btn{background:#d0eded !important;border-color:#a0d0d0 !important;color:#007a7a !important;}
body[data-theme="light"] .act-btn svg{stroke:#007a7a !important;}
body[data-theme="light"] .act-btn:hover{background:rgba(0,150,150,0.15) !important;border-color:#007a7a !important;}
body[data-theme="light"] .div .dl{background:#c0e0e0 !important;}
body[data-theme="light"] .div .dt{color:#b0d0d0 !important;}
body[data-theme="light"] .modal{background:#ffffff !important;border-color:#80cccc !important;box-shadow:0 20px 60px rgba(0,80,80,0.2) !important;}
body[data-theme="light"] .modal-title{color:#003333 !important;font-weight:800 !important;}
body[data-theme="light"] .modal-sub{color:#666666 !important;}
body[data-theme="light"] .step-text{color:#111111 !important;}
body[data-theme="light"] .step-num{background:rgba(0,100,100,0.12) !important;border-color:rgba(0,100,100,0.3) !important;color:#005555 !important;font-weight:700 !important;}
body[data-theme="light"] .ex-card{background:#f0fafa !important;border-color:#80cccc !important;}
body[data-theme="light"] .ex-role{color:#003333 !important;font-weight:700 !important;}
body[data-theme="light"] .ex-score{color:#336666 !important;}
body[data-theme="light"] .ex-chip.ex-have{background:rgba(0,100,100,0.1) !important;border-color:rgba(0,100,100,0.3) !important;color:#005555 !important;}
body[data-theme="light"] .ex-chip.ex-miss{background:rgba(160,0,0,0.07) !important;border-color:rgba(160,0,0,0.25) !important;color:#aa0000 !important;}
body[data-theme="light"] .about-text{color:#111111 !important;}
body[data-theme="light"] .about-tag{background:rgba(0,100,100,0.1) !important;border-color:rgba(0,100,100,0.25) !important;color:#005555 !important;}
body[data-theme="light"] .toggle-wrap{background:#ffffff !important;border-color:#80cccc !important;}
body[data-theme="light"] .tbtn{color:#336666 !important;}
body[data-theme="light"] .d-label{color:#888888 !important;}
body[data-theme="light"] .dropdown{background:#ffffff !important;border-color:#80cccc !important;box-shadow:0 8px 30px rgba(0,80,80,0.15) !important;}
body[data-theme="light"] .tbtn:hover{background:rgba(0,100,100,0.08) !important;color:#003333 !important;}
body[data-theme="light"] .tbtn.active{background:rgba(0,100,100,0.12) !important;color:#003333 !important;border-color:rgba(0,100,100,0.25) !important;}
#sg{width:440px;height:440px;position:fixed;border-radius:50%;pointer-events:none;z-index:1;transform:translate(-50%,-50%);background:radial-gradient(circle,rgba(0,255,200,0.12) 0%,rgba(0,188,212,0.04) 45%,transparent 70%);transition:background 0.35s;top:0;left:0;}
#mg{width:46px;height:46px;position:fixed;pointer-events:none;z-index:9999;opacity:0;transition:opacity 0.15s ease,transform 0.15s ease;top:0;left:0;transform:translate(-50%,-50%) scale(0.5);}
#mg.show{opacity:1;transform:translate(-50%,-50%) scale(1);}
#mg svg{width:100%;height:100%;filter:drop-shadow(0 0 14px #00ffe7) drop-shadow(0 0 6px #00ffe7);}
.bg{position:fixed;inset:0;z-index:0;background-image:radial-gradient(circle,#0a2a2a 1px,transparent 1px);background-size:28px 28px;opacity:0.4;pointer-events:none;}
.wrap{position:relative;z-index:2;padding:32px 40px 24px;}
nav{display:flex;align-items:center;justify-content:space-between;margin-bottom:48px;}
.logo{font-size:20px;font-weight:700;letter-spacing:-0.5px;}
.lr{color:#00ffe7;}.li{color:#00bcd4;}
.nav-links{display:flex;gap:26px;}
.nav-links{display:flex;align-items:center;gap:26px;}
.nav-links a{font-size:13px;color:#1a5050;text-decoration:none;cursor:pointer;transition:color 0.2s;}
.nav-links a:hover{color:#00ffe7;}
.menu-wrap{position:relative;}
.dots-btn{background:rgba(0,255,200,0.06);border:1px solid #0a3a3a;border-radius:10px;color:#1a5050;font-size:16px;width:36px;height:36px;cursor:pointer;transition:all 0.2s;display:flex;align-items:center;justify-content:center;letter-spacing:1px;}
.dots-btn:hover{border-color:#00ffe7;color:#00ffe7;background:rgba(0,255,200,0.1);}
.dropdown{position:absolute;top:calc(100% + 8px);right:0;background:#0a1414;border:1px solid #0a3a3a;border-radius:14px;padding:6px;min-width:150px;display:none;z-index:9999;animation:drop 0.15s ease;box-shadow:0 8px 30px rgba(0,0,0,0.5);}
@keyframes drop{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
.dropdown.open{display:block;}
.d-label{font-family:'Space Mono',monospace;font-size:9px;color:#1a4040;letter-spacing:0.12em;text-transform:uppercase;padding:6px 10px 4px;}
.tbtn{display:flex;align-items:center;gap:8px;width:100%;font-family:'Space Mono',monospace;font-size:11px;letter-spacing:0.04em;padding:9px 12px;border-radius:8px;border:none;background:transparent;color:#1a6060;cursor:pointer;transition:all 0.2s;text-align:left;}
.tbtn:hover{background:rgba(0,255,200,0.08);color:#00ffe7;}
.tbtn.active{background:rgba(0,255,200,0.12);color:#00ffe7;}
.tbtn .icon{font-size:13px;width:18px;text-align:center;}
.tbtn .check{margin-left:auto;opacity:0;color:#00ffe7;font-size:11px;}
.tbtn.active .check{opacity:1;}
.hero{text-align:center;margin-bottom:36px;}
.badge{display:inline-flex;align-items:center;gap:7px;background:rgba(0,255,200,0.07);border:1px solid rgba(0,255,200,0.18);color:#00ffe7;padding:5px 16px;border-radius:100px;font-family:'Space Mono',monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:22px;}
.dot{width:6px;height:6px;background:#00ffe7;border-radius:50%;box-shadow:0 0 8px #00ffe7;animation:blink 2s infinite;display:inline-block;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.25}}
h1{font-size:52px;font-weight:700;letter-spacing:-3px;line-height:1.02;margin-bottom:12px;}
.w1{color:#00ffe7;transition:none!important;}.w2{color:#00bcd4;transition:none!important;}
.sub{font-size:14px;color:#1a5050;line-height:1.75;max-width:400px;margin:0 auto 32px;font-weight:300;transition:none!important;}
.btn-row{display:flex;align-items:center;justify-content:center;gap:10px;margin-bottom:20px;flex-wrap:wrap;}
.act-btn{position:relative;display:flex;align-items:center;gap:8px;padding:10px 18px;border-radius:12px;border:1px solid #0a2a2a;background:#0a1010;color:#1a5050;font-family:'Space Mono',monospace;font-size:11px;letter-spacing:0.05em;cursor:pointer;transition:all 0.2s;white-space:nowrap;}
.act-btn svg{width:16px;height:16px;stroke:#1a5050;transition:stroke 0.2s;flex-shrink:0;}
.act-btn:hover{border-color:rgba(0,255,200,0.5);background:rgba(0,255,200,0.07);color:#00ffe7;box-shadow:0 0 16px rgba(0,255,200,0.1);}
.act-btn:hover svg{stroke:#00ffe7;filter:drop-shadow(0 0 4px rgba(0,255,200,0.7));}
.act-btn:active{transform:scale(0.97);}
#file-input{display:none;}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%) translateY(20px);background:#0d1a1a;border:1px solid #00ffe7;color:#00ffe7;padding:10px 20px;border-radius:10px;font-family:'Space Mono',monospace;font-size:11px;opacity:0;transition:all 0.3s;z-index:999;pointer-events:none;}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0);}
.div{display:flex;align-items:center;gap:12px;}
.dl{flex:1;height:1px;background:#0a1a1a;}
.dt{font-family:'Space Mono',monospace;font-size:10px;color:#0a2a2a;letter-spacing:0.12em;text-transform:uppercase;}
/* MODALS */
.modal-overlay{position:fixed;top:-9999px;left:-9999px;right:-9999px;bottom:-9999px;width:300vw;height:300vh;margin:-100vh -100vw;background:rgba(0,0,0,0.75);backdrop-filter:blur(6px);z-index:100;display:none;align-items:center;justify-content:center;}
.modal-overlay.show{display:flex;}
.modal{background:#0a1414;border:1px solid #0a3a3a;border-radius:20px;padding:32px;max-width:500px;width:90%;max-height:80vh;overflow-y:auto;position:relative;animation:mIn 0.25s ease;}
@keyframes mIn{from{opacity:0;transform:scale(0.95)}to{opacity:1;transform:scale(1)}}
.modal-close{position:absolute;top:16px;right:16px;background:transparent;border:none;color:#1a5050;font-size:20px;cursor:pointer;transition:color 0.2s;line-height:1;}
.modal-close:hover{color:#00ffe7;}
.modal-title{font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:#00ffe7;margin-bottom:6px;letter-spacing:-0.5px;}
.modal-sub{font-family:'Space Mono',monospace;font-size:9px;color:#1a5050;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:24px;}
.step{display:flex;gap:14px;margin-bottom:18px;align-items:flex-start;}
.step-num{width:28px;height:28px;border-radius:8px;background:rgba(0,255,200,0.1);border:1px solid rgba(0,255,200,0.2);color:#00ffe7;font-family:'Space Mono',monospace;font-size:11px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.step-text{font-family:'Space Grotesk',sans-serif;font-size:13px;color:#1a8080;line-height:1.6;}
.step-text strong{color:#00ffe7;font-weight:600;}
.ex-card{background:#060e0e;border:1px solid #0a2a2a;border-radius:12px;padding:16px;margin-bottom:10px;}
.ex-role{font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;color:#00ffe7;margin-bottom:4px;}
.ex-score{font-family:'Space Mono',monospace;font-size:11px;color:#1a6060;margin-bottom:8px;}
.ex-chips{display:flex;gap:5px;flex-wrap:wrap;}
.ex-chip{font-family:'Space Mono',monospace;font-size:9px;padding:3px 9px;border-radius:100px;}
.ex-have{background:rgba(0,255,200,0.08);border:1px solid rgba(0,255,200,0.2);color:#00ffe7;}
.ex-miss{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.2);color:#f87171;}
.about-tag{display:inline-block;background:rgba(0,255,200,0.08);border:1px solid rgba(0,255,200,0.2);color:#00ffe7;font-family:'Space Mono',monospace;font-size:9px;padding:3px 10px;border-radius:100px;margin:3px 3px 3px 0;}
.about-text{font-family:'Space Grotesk',sans-serif;font-size:13px;color:#1a8080;line-height:1.7;margin-bottom:14px;}
</style>
</head>
<body>
<div class="bg"></div>
<div id="sg"></div>
<div id="mg">
  <svg viewBox="0 0 24 24" fill="none">
    <circle cx="11" cy="11" r="7.5" stroke="#00ffe7" stroke-width="2"/>
    <line x1="16.5" y1="16.5" x2="21" y2="21" stroke="#00ffe7" stroke-width="2" stroke-linecap="round"/>
  </svg>
</div>
<div class="toast" id="toast"></div>
<input type="file" id="file-input" accept=".pdf"/>

<div class="wrap">
  <nav>
    <div class="logo"><span class="lr">Resume</span><span class="li">IQ</span></div>
    <div class="nav-links">
      <a onclick="sendModal('how')">How it works</a>
      <a onclick="sendModal('examples')">Examples</a>
      <a onclick="sendModal('about')">About</a>
      <div class="menu-wrap">
        <button class="dots-btn" onclick="toggleMenu()">⋯</button>
        <div class="dropdown" id="dropdown">
          <div class="d-label">Appearance</div>
          <button class="tbtn" id="btn-dark" onclick="setTheme('dark')">
            <span class="icon">🌑</span> Dark <span class="check">✓</span>
          </button>
          <button class="tbtn" id="btn-light" onclick="setTheme('light')">
            <span class="icon">☀️</span> Light <span class="check">✓</span>
          </button>
          <button class="tbtn" id="btn-system" onclick="setTheme('system')">
            <span class="icon">💻</span> System <span class="check">✓</span>
          </button>
        </div>
      </div>
    </div>
  </nav>

  <!-- MODALS -->
  <div class="modal-overlay" id="modal-how">
    <div class="modal">
      <button class="modal-close" onclick="closeModal('how')">✕</button>
      <div class="modal-title">How it works</div>
      <div class="modal-sub">4 simple steps</div>
      <div class="step"><div class="step-num">01</div><div class="step-text"><strong>Upload your resume</strong> — PDF is parsed and all skills, keywords and content are extracted instantly.</div></div>
      <div class="step"><div class="step-num">02</div><div class="step-text"><strong>Paste a job description</strong> — Copy any job posting from LinkedIn, Naukri, or any job board.</div></div>
      <div class="step"><div class="step-num">03</div><div class="step-text"><strong>Get your analysis</strong> — AI calculates match score, checks ATS compatibility, finds skill gaps and gives a personalized learning roadmap.</div></div>
      <div class="step"><div class="step-num">04</div><div class="step-text"><strong>Improve and reapply</strong> — Use suggestions and roadmap to upskill and re-analyse until your score is above 65%.</div></div>
    </div>
  </div>

  <div class="modal-overlay" id="modal-examples">
    <div class="modal">
      <button class="modal-close" onclick="closeModal('examples')">✕</button>
      <div class="modal-title">Example Results</div>
      <div class="modal-sub">See what the analysis looks like</div>
      <div class="ex-card">
        <div class="ex-role">Python Developer @ Startup</div>
        <div class="ex-score">Match Score: 72% · Strong Match 🎯</div>
        <div class="ex-chips"><span class="ex-chip ex-have">✓ python</span><span class="ex-chip ex-have">✓ flask</span><span class="ex-chip ex-have">✓ sql</span><span class="ex-chip ex-have">✓ git</span><span class="ex-chip ex-miss">✗ docker</span><span class="ex-chip ex-miss">✗ aws</span></div>
      </div>
      <div class="ex-card">
        <div class="ex-role">ML Engineer @ Product Company</div>
        <div class="ex-score">Match Score: 45% · Moderate Match ⚡</div>
        <div class="ex-chips"><span class="ex-chip ex-have">✓ python</span><span class="ex-chip ex-have">✓ pandas</span><span class="ex-chip ex-miss">✗ tensorflow</span><span class="ex-chip ex-miss">✗ pytorch</span><span class="ex-chip ex-miss">✗ kubernetes</span></div>
      </div>
      <div class="ex-card">
        <div class="ex-role">Frontend Developer @ MNC</div>
        <div class="ex-score">Match Score: 85% · Strong Match 🎯</div>
        <div class="ex-chips"><span class="ex-chip ex-have">✓ react</span><span class="ex-chip ex-have">✓ javascript</span><span class="ex-chip ex-have">✓ typescript</span><span class="ex-chip ex-have">✓ css</span><span class="ex-chip ex-miss">✗ graphql</span></div>
      </div>
    </div>
  </div>

  <div class="modal-overlay" id="modal-about">
    <div class="modal">
      <button class="modal-close" onclick="closeModal('about')">✕</button>
      <div class="modal-title">About ResumeIQ</div>
      <div class="modal-sub">AI-powered career intelligence</div>
      <div class="about-text">ResumeIQ is a free AI-powered tool that helps students and job seekers understand exactly how well their resume matches any job description — before they apply.</div>
      <div class="about-text">Built with Python, NLP and Streamlit. Uses TF-IDF cosine similarity for matching and rule-based ATS scoring.</div>
      <div class="about-text"><strong style="color:#003333;">Tech Stack</strong></div>
      <div style="margin-bottom:14px;"><span class="about-tag">Python</span><span class="about-tag">Streamlit</span><span class="about-tag">Scikit-learn</span><span class="about-tag">PyPDF2</span><span class="about-tag">TF-IDF</span><span class="about-tag">Cosine Similarity</span></div>
      <div class="about-text"><strong style="color:#003333;">Features</strong></div>
      <div><span class="about-tag">Match Score</span><span class="about-tag">ATS Checker</span><span class="about-tag">Skill Gap Analysis</span><span class="about-tag">AI Suggestions</span><span class="about-tag">Learning Roadmap</span><span class="about-tag">Score Breakdown</span><span class="about-tag">Top Keywords</span><span class="about-tag">Resume Stats</span></div>
    </div>
  </div>

  <div class="hero">
    <div class="badge"><span class="dot"></span>AI Career Intelligence</div>
    <h1><span class="w1">Know your </span><span class="w2">exact</span><br><span class="w1">job fit </span><span class="w2">score</span></h1>
    <p class="sub">Upload your resume, paste a job description — get an instant match score with skill insights.</p>
  </div>

  <div class="btn-row">
    <button class="act-btn mag-trigger" data-c="#00ffe7" onclick="triggerUpload()">
      <svg fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>Upload Resume
    </button>
    <button class="act-btn mag-trigger" data-c="#fbbf24" onclick="pasteJD()">
      <svg fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>Paste JD
    </button>
    <button class="act-btn mag-trigger" data-c="#4ade80" onclick="copyResults()">
      <svg fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>Copy Results
    </button>
    <button class="act-btn mag-trigger" data-c="#fb7185" onclick="clearAll()">
      <svg fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>Clear All
    </button>
    <button class="act-btn mag-trigger" data-c="#a78bfa" onclick="shareApp()">
      <svg fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"/></svg>Share
    </button>
  </div>

  <div class="div"><div class="dl"></div><div class="dt">or fill in below</div><div class="dl"></div></div>
</div>

<script>
// Sync theme from localStorage on hero load
(function(){
  const t = localStorage.getItem('resumeiq-theme') || 'dark';
  if(t === 'light') document.body.setAttribute('data-theme','light');
  else document.body.setAttribute('data-theme','dark');
  // Watch for changes
  window.addEventListener('storage', e => {
    if(e.key==='resumeiq-theme'){
      document.body.setAttribute('data-theme', e.newValue || 'dark');
    }
  });
})();

// ── Modal via postMessage ──
function sendModal(type){
  try {
    window.parent.openGlobalModal(type);
  } catch(e) {
    window.parent.postMessage({type:'openModal', modal: type}, '*');
  }
}

// ── Theme functions ──
function toggleMenu(){
  document.getElementById('dropdown').classList.toggle('open');
}
document.addEventListener('click',function(e){
  if(!e.target.closest('.menu-wrap')) {
    const dd = document.getElementById('dropdown');
    if(dd) dd.classList.remove('open');
  }
});

function applyTheme(theme){
  const root = window.parent.document.documentElement;
  const stApp = window.parent.document.querySelector('.stApp');
  const iframes = window.parent.document.querySelectorAll('iframe');

  if(theme==='light'){
    root.setAttribute('data-theme','light');
    if(stApp) stApp.style.background='#eaf6f6';
    document.body.setAttribute('data-theme','light');
    document.body.style.background='#eaf6f6';
    // dropdown light style
    const dd=document.getElementById('dropdown');
    if(dd){dd.style.background='#d8f0f0';dd.style.borderColor='#90cccc';}
    iframes.forEach(f=>{try{f.contentDocument.body.setAttribute('data-theme','light');f.contentDocument.body.style.background='#eaf6f6';}catch(e){}});
  } else if(theme==='dark'){
    root.setAttribute('data-theme','dark');
    if(stApp) stApp.style.background='#060608';
    document.body.setAttribute('data-theme','dark');
    document.body.style.background='#060608';
    const dd=document.getElementById('dropdown');
    if(dd){dd.style.background='#0a1414';dd.style.borderColor='#0a3a3a';}
    iframes.forEach(f=>{try{f.contentDocument.body.setAttribute('data-theme','dark');f.contentDocument.body.style.background='#060608';}catch(e){}});
  } else {
    const sys=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
    applyTheme(sys); return;
  }
  localStorage.setItem('resumeiq-theme',theme);
  document.querySelectorAll('.tbtn').forEach(b=>b.classList.remove('active'));
  const btn=document.getElementById('btn-'+theme);
  if(btn) btn.classList.add('active');
  const dd=document.getElementById('dropdown');
  if(dd) dd.classList.remove('open');
}

function setTheme(t){ applyTheme(t); }

// Load on startup
(function(){
  const t=localStorage.getItem('resumeiq-theme')||'dark';
  document.body.setAttribute('data-theme',t);
  document.querySelectorAll('.tbtn').forEach(b=>b.classList.remove('active'));
  const btn=document.getElementById('btn-'+t);
  if(btn) btn.classList.add('active');
  applyTheme(t);
})();

window.addEventListener('storage',e=>{
  if(e.key==='resumeiq-theme') applyTheme(e.newValue||'dark');
});
// ──────────────────────

const sg=document.getElementById('sg'),mg=document.getElementById('mg');
const mgC=mg.querySelector('circle'),mgL=mg.querySelector('line');
let mx=window.innerWidth/2,my=window.innerHeight/2,gx=mx,gy=my;
document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;mg.style.left=mx+'px';mg.style.top=my+'px';});
(function loop(){gx+=(mx-gx)*0.07;gy+=(my-gy)*0.07;sg.style.left=gx+'px';sg.style.top=gy+'px';requestAnimationFrame(loop);})();
// Color maps — dark uses vivid neon, light uses rich saturated versions
const darkColors = {
  '#00ffe7': '#00ffe7',
  '#fbbf24': '#fbbf24',
  '#4ade80': '#4ade80',
  '#fb7185': '#fb7185',
  '#a78bfa': '#a78bfa',
};
const lightColors = {
  '#00ffe7': '#00a896',
  '#fbbf24': '#d97706',
  '#4ade80': '#16a34a',
  '#fb7185': '#e11d48',
  '#a78bfa': '#7c3aed',
};

function isLight(){ return document.body.getAttribute('data-theme')==='light'; }

document.querySelectorAll('.mag-trigger').forEach(btn=>{
  const key = btn.dataset.c;
  btn.addEventListener('mouseenter',()=>{
    const c = isLight() ? (lightColors[key]||key) : (darkColors[key]||key);
    const glowAlpha = isLight() ? '40' : '33';

    // show magnifier with matching color
    mg.classList.add('show');
    mgC.setAttribute('stroke', c);
    mgL.setAttribute('stroke', c);
    mg.querySelector('svg').style.filter = `drop-shadow(0 0 14px ${c}) drop-shadow(0 0 8px ${c})`;

    // spotlight glow
    sg.style.background = `radial-gradient(circle,${c}${glowAlpha} 0%,transparent 70%)`;

    // change headline + subtitle text color
    document.querySelectorAll('.w1').forEach(el => {
      el.style.setProperty('color', c, 'important');
      el.style.setProperty('-webkit-text-fill-color', c, 'important');
    });
    document.querySelectorAll('.w2').forEach(el => {
      el.style.setProperty('color', c, 'important');
      el.style.setProperty('-webkit-text-fill-color', c, 'important');
    });
    const sub = document.querySelector('.sub');
    if(sub) sub.style.setProperty('color', c + 'bb', 'important');

    // button glow border
    btn.style.borderColor = c;
    btn.style.color = c;
    btn.style.boxShadow = `0 0 22px ${c}55`;
    btn.querySelector('svg').style.stroke = c;
  });

  btn.addEventListener('mouseleave',()=>{
    mg.classList.remove('show');
    btn.style.borderColor = '';
    btn.style.color = '';
    btn.style.boxShadow = '';
    if(btn.querySelector('svg')) btn.querySelector('svg').style.stroke = '';

    if(isLight()){
      sg.style.background = 'radial-gradient(circle,rgba(0,150,150,0.1) 0%,transparent 70%)';
      document.querySelectorAll('.w1').forEach(el => {
        el.style.setProperty('color','#007a7a','important');
        el.style.setProperty('-webkit-text-fill-color','#007a7a','important');
      });
      document.querySelectorAll('.w2').forEach(el => {
        el.style.setProperty('color','#009999','important');
        el.style.setProperty('-webkit-text-fill-color','#009999','important');
      });
      const sub = document.querySelector('.sub');
      if(sub) sub.style.setProperty('color','#aacccc','important');
    } else {
      sg.style.background = 'radial-gradient(circle,rgba(0,255,200,0.12) 0%,rgba(0,188,212,0.04) 45%,transparent 70%)';
      document.querySelectorAll('.w1').forEach(el => {
        el.style.setProperty('color','#00ffe7','important');
        el.style.setProperty('-webkit-text-fill-color','#00ffe7','important');
      });
      document.querySelectorAll('.w2').forEach(el => {
        el.style.setProperty('color','#00bcd4','important');
        el.style.setProperty('-webkit-text-fill-color','#00bcd4','important');
      });
      const sub = document.querySelector('.sub');
      if(sub) sub.style.setProperty('color','#1a5050','important');
    }
  });
});
function showToast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2500);}
function triggerUpload(){document.getElementById('file-input').click();}
document.getElementById('file-input').addEventListener('change',function(){if(this.files[0])showToast('✓ '+this.files[0].name+' — scroll down to upload');});
async function pasteJD(){try{const text=await navigator.clipboard.readText();if(text)showToast('✓ Clipboard ready! Paste in the Job Description box below');else showToast('Copy a job description first, then click Paste JD');}catch(e){showToast('📋 Copy a job description then paste in the box below');}}
function copyResults(){try{navigator.clipboard.writeText(window.top.location.href);showToast('✓ App link copied!');}catch(e){showToast('Copy the URL from your browser to share');}}
function clearAll(){showToast('🗑 Clearing...');setTimeout(()=>window.top.location.reload(),800);}
function shareApp(){try{navigator.clipboard.writeText(window.top.location.href);showToast('✓ App link copied! Share with anyone');}catch(e){showToast('Copy the URL from your browser to share');}}
function openModal(id){document.getElementById('modal-'+id).classList.add('show');}
function closeModal(id){document.getElementById('modal-'+id).classList.remove('show');}
document.addEventListener('click',e=>{if(e.target.classList.contains('modal-overlay'))e.target.classList.remove('show');});
</script>
</body>
</html>
""", height=600, scrolling=False)

# ── Inputs ───────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="medium")
with col1:
    st.markdown('<div class="sec-label">01 / Your Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("resume", type="pdf", label_visibility="collapsed")
    if uploaded_file:
        st.markdown(f'<div class="ok-file">✓ {uploaded_file.name}</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="sec-label">02 / Job Description</div>', unsafe_allow_html=True)
    jd = st.text_area("jd", height=160, placeholder="Paste the full job description here...", label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("⚡  Analyse Resume", use_container_width=True)

if run:
    if uploaded_file and jd:
        with st.spinner("Analysing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)

            # FIX 3: Validate PDF content
            if not resume_text or len(resume_text) < 100:
                st.markdown("""
                <div class="error-box">
                  ⚠️ Could not extract text from this PDF.<br><br>
                  This usually happens with <strong>scanned or image-based PDFs</strong>.
                  Please try a text-based PDF, or copy-paste your resume content manually.
                </div>
                """, unsafe_allow_html=True)
                st.stop()

            score        = calculate_match(resume_text, jd)
            have         = extract_skills(resume_text)
            missing      = find_missing(resume_text, jd)
            label, desc  = score_label(score)
            color        = score_color(score)
            ats, ats_tips = ats_score(resume_text, uploaded_file.name)
            tips         = improvement_suggestions(have, missing, score)
            ats_color    = "#00ffe7" if ats>=75 else "#fbbf24" if ats>=55 else "#f87171"
            top_kw       = extract_top_keywords(jd)
            words, pages, _ = resume_stats(resume_text)

        circ   = 2*3.14159*40
        offset = circ*(1-min(score,100)/100)
        have_html = "".join(f'<span class="chip have">✓ {s}</span>' for s in have) or '<span style="color:#1a5050;font-size:12px;">No matching skills detected</span>'
        miss_html = "".join(f'<span class="chip miss">✗ {s}</span>' for s in missing) or '<span style="color:#00ffe7;font-size:12px;">🎉 No skill gaps found!</span>'

        # ── Match Score ──
        st.markdown(f"""
        <div class="res-wrap">
          <div class="score-card">
            <div class="ring">
              <svg width="96" height="96" viewBox="0 0 96 96">
                <circle cx="48" cy="48" r="40" fill="none" stroke="#0a2a2a" stroke-width="7"/>
                <circle cx="48" cy="48" r="40" fill="none" stroke="{color}" stroke-width="7"
                  stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}" stroke-linecap="round"
                  style="transition:stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1);"/>
              </svg>
              <div class="ring-num">{score}%</div>
            </div>
            <div class="score-right">
              <div class="score-verdict">{label}</div>
              <div class="score-desc">{desc}</div>
              <div class="score-explain">Score = keyword overlap between your resume and the job description using TF-IDF cosine similarity. Higher overlap = higher score.</div>
              <div class="chips">{have_html}</div>
            </div>
          </div>
          <div class="score-card" style="flex-direction:column;align-items:flex-start;">
            <div style="font-family:Space Mono,monospace;font-size:9px;color:#1a5050;letter-spacing:0.16em;text-transform:uppercase;margin-bottom:12px;">Skills to Acquire</div>
            <div class="chips">{miss_html}</div>
          </div>
          <div class="stats">
            <div class="stat"><div class="stat-n" style="color:#00ffe7;">{len(have)}</div><div class="stat-l">Matched</div></div>
            <div class="stat"><div class="stat-n" style="color:#f87171;">{len(missing)}</div><div class="stat-l">Missing</div></div>
            <div class="stat"><div class="stat-n" style="color:#a78bfa;">{len(have)+len(missing)}</div><div class="stat-l">Required</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── FIX 4: Resume Stats ──
        page_color = "#00ffe7" if 0.8 <= pages <= 2 else "#fbbf24"
        st.markdown(f"""
        <div class="feat-card">
          <div class="feat-title">📄 Resume Stats</div>
          <span class="resume-stat">📝 {words} words</span>
          <span class="resume-stat" style="color:{page_color};">📄 ~{pages} pages</span>
          <span class="resume-stat">🔤 {len(have)} skills detected</span>
          {"<br><br><span style='font-family:Space Grotesk,sans-serif;font-size:12px;color:#fbbf24;'>⚠️ Resume seems short — add more detail to improve ATS score</span>" if pages < 0.8 else ""}
          {"<br><br><span style='font-family:Space Grotesk,sans-serif;font-size:12px;color:#fbbf24;'>⚠️ Resume might be too long — consider trimming to under 2 pages</span>" if pages > 2 else ""}
        </div>
        """, unsafe_allow_html=True)

        # ── Score Breakdown ──
        skill_score   = round(min((len(have)/max(len(have)+len(missing),1))*100,100))
        keyword_score = min(round(score*1.2),100)
        content_score = min(round(words/7),100)
        st.markdown(f"""
        <div class="feat-card">
          <div class="feat-title">📊 Score Breakdown</div>
          <div class="breakdown-row">
            <div class="breakdown-label"><span>Skills Match</span><span style="color:#00ffe7;">{skill_score}%</span></div>
            <div class="breakdown-track"><div class="breakdown-fill" style="width:{skill_score}%;background:linear-gradient(90deg,#00ffe7,#00bcd4);"></div></div>
          </div>
          <div class="breakdown-row">
            <div class="breakdown-label"><span>Keyword Relevance</span><span style="color:#a78bfa;">{keyword_score}%</span></div>
            <div class="breakdown-track"><div class="breakdown-fill" style="width:{keyword_score}%;background:linear-gradient(90deg,#a78bfa,#7c3aed);"></div></div>
          </div>
          <div class="breakdown-row">
            <div class="breakdown-label"><span>Content Depth</span><span style="color:#fbbf24;">{content_score}%</span></div>
            <div class="breakdown-track"><div class="breakdown-fill" style="width:{content_score}%;background:linear-gradient(90deg,#fbbf24,#f59e0b);"></div></div>
          </div>
          <div class="breakdown-row">
            <div class="breakdown-label"><span>ATS Compatibility</span><span style="color:{ats_color};">{ats}%</span></div>
            <div class="breakdown-track"><div class="breakdown-fill" style="width:{ats}%;background:{ats_color};"></div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── FIX 5: Top Keywords from JD ──
        kw_html = "".join(f'<span class="kw-chip">{k}</span>' for k in top_kw)
        st.markdown(f"""
        <div class="feat-card">
          <div class="feat-title">🔑 Top Keywords from Job Description</div>
          <div style="margin-bottom:8px;font-family:Space Grotesk,sans-serif;font-size:12px;color:#1a5050;">Add these words to your resume to improve your match score:</div>
          <div>{kw_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── ATS Score ──
        ats_label    = "ATS Friendly ✓" if ats>=75 else "Needs Improvement" if ats>=55 else "Poor ATS Score"
        ats_tips_html= "".join(f'<div class="tip-item"><div class="tip-dot" style="background:{ats_color};"></div><div class="tip-text">{t}</div></div>' for t in ats_tips)
        st.markdown(f"""
        <div class="feat-card">
          <div class="feat-title">🤖 ATS Score Checker</div>
          <div class="ats-wrap">
            <div class="ats-num" style="color:{ats_color};">{ats}%</div>
            <div class="ats-right">
              <div class="ats-label" style="color:{ats_color};">{ats_label}</div>
              <div class="ats-bar"><div class="ats-fill" style="width:{ats}%;background:{ats_color};"></div></div>
            </div>
          </div>
          {ats_tips_html}
        </div>
        """, unsafe_allow_html=True)

        # ── AI Suggestions ──
        tips_html = "".join(f'<div class="tip-item"><div class="tip-dot" style="background:#00ffe7;"></div><div class="tip-text">{t}</div></div>' for t in tips)
        st.markdown(f"""
        <div class="feat-card">
          <div class="feat-title">💡 AI Improvement Suggestions</div>
          {tips_html}
        </div>
        """, unsafe_allow_html=True)

        # ── Learning Roadmap ──
        if missing:
            road_html = "".join(
                f'<div class="road-item"><div class="road-skill">✗ {s}</div><a class="road-link" href="{ROADMAP[s][1]}" target="_blank">Learn {ROADMAP[s][0]} →</a></div>'
                for s in missing[:8] if s in ROADMAP
            )
            if road_html:
                st.markdown(f"""
                <div class="feat-card">
                  <div class="feat-title">🗺️ Learning Roadmap for Missing Skills</div>
                  {road_html}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#fff3cd;border:1.5px solid #e0a800;border-radius:12px;padding:14px 20px;margin-top:8px;display:flex;align-items:center;gap:10px;">
          <span style="font-size:18px;">⚠️</span>
          <span style="color:#7a4800;font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;">Please upload a resume PDF and paste a job description.</span>
        </div>
        """, unsafe_allow_html=True)