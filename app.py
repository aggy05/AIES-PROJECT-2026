import streamlit as st
import pdfplumber
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResuMatch – AI Hiring Platform",
    page_icon="🎯",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background: #0d0d0d; color: #f0ece4; }
[data-testid="stSidebar"] { background: #111111; border-right: 1px solid #222; }
[data-testid="stSidebar"] * { color: #f0ece4 !important; }

.card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 24px; margin-bottom: 16px; }
.card-accent { background: #1a1a1a; border: 1px solid #c8f04a; border-radius: 12px; padding: 24px; margin-bottom: 16px; }

.score-big { font-family: 'Syne', sans-serif; font-size: 72px; font-weight: 800; line-height: 1; }
.score-good { color: #c8f04a; }
.score-mid  { color: #f0b429; }
.score-low  { color: #f05a4a; }

.pill-matched { display:inline-block; background:#1e3a1e; border:1px solid #4caf50; color:#a5d6a7; border-radius:20px; padding:4px 14px; margin:4px; font-size:13px; }
.pill-missing { display:inline-block; background:#3a1e1e; border:1px solid #f05a4a; color:#ef9a9a; border-radius:20px; padding:4px 14px; margin:4px; font-size:13px; }
.pill-neutral { display:inline-block; background:#1e1e2e; border:1px solid #5c6bc0; color:#9fa8da; border-radius:20px; padding:4px 14px; margin:4px; font-size:13px; }

.hero { background:linear-gradient(135deg,#1a1a1a 0%,#0d0d0d 100%); border:1px solid #2a2a2a; border-radius:16px; padding:40px; margin-bottom:32px; text-align:center; }
.hero h1 { font-size:48px; font-weight:800; color:#f0ece4; margin-bottom:8px; }
.hero h1 span { color:#c8f04a; }
.hero p { color:#888; font-size:18px; }

.feedback-box { background:#141414; border-left:3px solid #c8f04a; border-radius:0 8px 8px 0; padding:14px 18px; margin-bottom:10px; }
.feedback-box p { color:#ccc; margin:0; font-size:14px; line-height:1.6; }

.suggestion-box { background:#141414; border-left:3px solid #5c6bc0; border-radius:0 8px 8px 0; padding:14px 18px; margin-bottom:8px; }
.suggestion-box strong { color:#9fa8da; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
.suggestion-box p { color:#aaa; margin:4px 0 0; font-size:14px; }

.score-mini-wrap { display:flex; gap:12px; margin-bottom:12px; }
.score-mini { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px; padding:12px 18px; flex:1; text-align:center; }
.score-mini .slabel { color:#555; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
.score-mini .svalue { font-family:'Syne',sans-serif; font-size:24px; font-weight:800; }

.divider { border:none; border-top:1px solid #222; margin:28px 0; }
.stProgress > div > div > div > div { background:#c8f04a; }
.stButton > button { background:#c8f04a; color:#0d0d0d; font-family:'Syne',sans-serif; font-weight:700; border:none; border-radius:8px; padding:10px 24px; }
.stButton > button:hover { background:#d4f75a; }
.stTextArea textarea, .stTextInput input { background:#1a1a1a !important; color:#f0ece4 !important; border:1px solid #2a2a2a !important; border-radius:8px !important; }
.stRadio label { color:#f0ece4 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
SKILLS_DATABASE = {
    "python": ["python"], "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "natural language processing": ["natural language processing", "nlp"],
    "data analysis": ["data analysis", "data analytics"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "sql": ["sql", "mysql", "postgresql"], "java": ["java"],
    "javascript": ["javascript", "js"], "html": ["html"], "css": ["css"],
    "excel": ["excel", "ms excel"], "communication": ["communication"],
    "teamwork": ["teamwork", "team player"], "problem solving": ["problem solving"],
    "data science": ["data science"], "tensorflow": ["tensorflow"], "keras": ["keras"],
    "pandas": ["pandas"], "numpy": ["numpy"], "power bi": ["power bi"],
    "tableau": ["tableau"], "c++": ["c++"], "r": [" r ", "r programming"],
}

SUGGESTIONS = {
    "python": "Learn Python at https://www.learnpython.org",
    "machine learning": "Take Andrew Ng's ML course on Coursera.",
    "deep learning": "Try Deep Learning Specialization on Coursera or fast.ai.",
    "natural language processing": "Explore HuggingFace NLP course at https://huggingface.co/learn",
    "data analysis": "Practice with pandas and real datasets on Kaggle.com",
    "artificial intelligence": "Start with AI For Everyone by Andrew Ng on Coursera.",
    "sql": "Practice SQL free at https://sqlbolt.com",
    "communication": "Add group projects or presentations to show communication skills.",
    "problem solving": "Practice on LeetCode or HackerRank and mention it on your resume.",
    "tensorflow": "Start with TensorFlow tutorials at https://www.tensorflow.org/tutorials",
    "keras": "Learn Keras at https://keras.io/getting_started",
    "pandas": "Practice pandas at https://pandas.pydata.org/docs",
    "numpy": "Learn NumPy at https://numpy.org/learn",
    "power bi": "Free Power BI learning at https://learn.microsoft.com/en-us/power-bi",
    "tableau": "Free Tableau training at https://www.tableau.com/learn/training",
    "data science": "Try IBM Data Science Professional Certificate on Coursera.",
    "excel": "Free Excel training at https://support.microsoft.com/en-us/excel",
    "java": "Learn Java at https://www.learnjavaonline.org",
    "javascript": "Start JavaScript at https://javascript.info",
    "html": "Learn HTML at https://www.w3schools.com/html",
    "css": "Learn CSS at https://www.w3schools.com/css",
    "teamwork": "Highlight group projects or hackathons on your resume.",
    "c++": "Learn C++ at https://www.learncpp.com",
    "r": "Learn R at https://www.r-project.org",
}

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────
def extract_text(uploaded_file):
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        text = ""
        for page in pdf.pages:
            t = page.extract_text()
            text += t if t else ""
    return text

def extract_skills(text):
    tl = text.lower()
    return [skill for skill, syns in SKILLS_DATABASE.items() if any(s in tl for s in syns)]

def keyword_match(resume_skills, job_skills):
    matched = [s for s in resume_skills if s in job_skills]
    missing = [s for s in job_skills if s not in resume_skills]
    score = round(len(matched) / len(job_skills) * 100, 2) if job_skills else 0
    return matched, missing, score

def sem_score(resume_text, job_text):
    vec = TfidfVectorizer(stop_words='english')
    v = vec.fit_transform([job_text, resume_text])
    return round(cosine_similarity(v[0], v[1])[0][0] * 100, 2)

def hybrid(k, s):
    return round(k * 0.6 + s * 0.4, 2)

def ai_feedback(resume_text, matched, missing, score):
    fb = []
    tl = resume_text.lower()
    if score >= 70:   fb.append("✅ STRONG MATCH: Your resume aligns well with the job. Focus on polishing the presentation.")
    elif score >= 40: fb.append("⚠️ MODERATE MATCH: Targeted improvements will significantly boost your chances.")
    else:             fb.append("❌ LOW MATCH: Your resume needs significant work to align with this role.")

    wc = len(resume_text.split())
    if wc < 200:    fb.append("📄 LENGTH: Resume is too short. Aim for 400–600 words covering skills, experience, and projects.")
    elif wc > 1000: fb.append("📄 LENGTH: Resume is quite long. Keep it concise — 1 page for freshers, 2 pages max.")
    else:           fb.append("📄 LENGTH: Resume length looks good.")

    if any(c.isdigit() for c in resume_text):
        fb.append("📊 IMPACT: Good — numbers/metrics found. These strengthen your impact statements.")
    else:
        fb.append("📊 IMPACT: No numbers found. Add quantifiable achievements like 'Improved accuracy by 20%'.")

    if "project" in tl:
        fb.append("🔨 PROJECTS: Project experience found. Clearly state the problem, tools used, and outcome for each.")
    else:
        fb.append("🔨 PROJECTS: No projects detected. Add 2–3 relevant projects to strengthen your resume.")

    if any(w in tl for w in ["bachelor","master","b.e","b.tech","mba","degree","university","college"]):
        fb.append("🎓 EDUCATION: Education section found. Mention your CGPA if it's above 7.0.")
    else:
        fb.append("🎓 EDUCATION: Education not clearly found. Ensure your degree, university, and year are visible.")

    if any(w in tl for w in ["certified","certification","certificate","coursera","udemy","nptel"]):
        fb.append("🏅 CERTIFICATIONS: Certifications found — great for credibility in technical roles.")
    else:
        fb.append("🏅 CERTIFICATIONS: No certifications found. Add relevant ones from Coursera, Udemy, or NPTEL.")

    return fb

def color_cls(s):
    return "score-good" if s >= 70 else "score-mid" if s >= 40 else "score-low"

def pills(skills, cls):
    return "".join(f'<span class="{cls}">{s.title()}</span>' for s in skills)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 ResuMatch")
    st.markdown("---")
    portal = st.radio("Nav", ["🏠 Home", "👤 Job Seeker Portal", "🏢 Employer Portal"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<small style='color:#444'>AI Resume Matching System<br>CSE (AIDS) — AIES Project 2026<br>Afiya · Agnes · Purvi</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────
if portal == "🏠 Home":
    st.markdown('<div class="hero"><h1>Resu<span>Match</span></h1><p>AI-powered Resume & Job Description Matching Platform</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card-accent"><h3 style="color:#c8f04a;font-family:Syne,sans-serif">👤 Job Seeker Portal</h3><p style="color:#aaa;margin-top:8px">Upload your resume + paste a job description → get hybrid match score, skill gap analysis, and AI-powered personalised feedback.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3 style="color:#f0ece4;font-family:Syne,sans-serif">🏢 Employer Portal</h3><p style="color:#aaa;margin-top:8px">Post a job description + upload multiple resumes → candidates are automatically ranked using keyword + semantic hybrid scoring.</p></div>', unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### How It Works")
    cols = st.columns(4)
    steps = [("📄","Upload Resume","PDF parsed automatically"),("🔍","Skill Extraction","Keyword + synonym matching"),("🧠","Semantic Analysis","TF-IDF cosine similarity"),("🎯","Results","Hybrid score + AI feedback")]
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f'<div class="card" style="text-align:center"><div style="font-size:28px">{icon}</div><h4 style="font-family:Syne,sans-serif;color:#f0ece4;margin:8px 0 4px">{title}</h4><p style="color:#555;font-size:13px">{desc}</p></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# JOB SEEKER PORTAL
# ─────────────────────────────────────────────
elif portal == "👤 Job Seeker Portal":
    st.markdown("## 👤 Job Seeker Portal")
    st.markdown("<p style='color:#666'>Upload your resume and a job description to get your full AI match report.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    cl, cr = st.columns(2, gap="large")
    with cl:
        st.markdown("#### 📄 Upload Resume (PDF)")
        rf = st.file_uploader("R", type=["pdf"], key="sk", label_visibility="collapsed")
        st.markdown("#### 📋 Job Description")
        jd = st.text_area("J", height=220, placeholder="Paste the job description here...", label_visibility="collapsed")
        go = st.button("🔍 Analyse My Resume", use_container_width=True)

    with cr:
        if go:
            if not rf: st.warning("Please upload your resume PDF.")
            elif not jd.strip(): st.warning("Please paste a job description.")
            else:
                with st.spinner("Analysing..."):
                    rt = extract_text(rf)
                    rs = extract_skills(rt)
                    js = extract_skills(jd)
                    if not js:
                        st.error("No skills found in job description.")
                    else:
                        matched, missing, ks = keyword_match(rs, js)
                        ss = sem_score(rt, jd)
                        hs = hybrid(ks, ss)
                        cc = color_cls(hs)

                        st.markdown(f'<div class="card" style="text-align:center;padding:28px"><p style="color:#555;font-size:12px;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px">Hybrid Match Score</p><div class="score-big {cc}">{hs}%</div><p style="color:#444;margin-top:8px;font-size:13px">{"🟢 Strong match!" if hs>=70 else "🟡 Moderate — room to grow." if hs>=40 else "🔴 Low match — focus on upskilling."}</p></div>', unsafe_allow_html=True)
                        st.progress(int(min(hs, 100)) / 100)

                        st.markdown(f'<div class="score-mini-wrap"><div class="score-mini"><div class="slabel">Keyword Score</div><div class="svalue" style="color:#c8f04a">{ks}%</div></div><div class="score-mini"><div class="slabel">Semantic Score</div><div class="svalue" style="color:#9fa8da">{ss}%</div></div></div>', unsafe_allow_html=True)

                        st.markdown("#### ✅ Matched Skills")
                        st.markdown(pills(matched, "pill-matched") if matched else "<span style='color:#666'>None found.</span>", unsafe_allow_html=True)

                        st.markdown("#### ❌ Missing Skills")
                        st.markdown(pills(missing, "pill-missing") if missing else "<span style='color:#4caf50'>🎉 All required skills present!</span>", unsafe_allow_html=True)

                        st.markdown("#### 🤖 AI Resume Feedback")
                        for pt in ai_feedback(rt, matched, missing, hs):
                            st.markdown(f'<div class="feedback-box"><p>{pt}</p></div>', unsafe_allow_html=True)

                        if missing:
                            st.markdown("#### 💡 How to Fill the Gaps")
                            for sk in missing:
                                tip = SUGGESTIONS.get(sk, f"Consider learning {sk.title()} through Coursera or Udemy.")
                                st.markdown(f'<div class="suggestion-box"><strong>{sk.upper()}</strong><p>{tip}</p></div>', unsafe_allow_html=True)

                        with st.expander("📋 All skills in your resume"):
                            st.markdown(pills(rs, "pill-neutral") if rs else "<span style='color:#666'>None found.</span>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card" style="text-align:center;padding:48px"><div style="font-size:48px">🎯</div><p style="color:#555;margin-top:12px">Upload your resume and paste a job description, then click <strong style="color:#c8f04a">Analyse My Resume</strong>.</p></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EMPLOYER PORTAL
# ─────────────────────────────────────────────
elif portal == "🏢 Employer Portal":
    st.markdown("## 🏢 Employer Portal")
    st.markdown("<p style='color:#666'>Post a job and upload resumes — candidates ranked by hybrid keyword + semantic score.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    cl, cr = st.columns(2, gap="large")
    with cl:
        st.markdown("#### 📋 Job Description")
        ejd = st.text_area("EJ", height=200, placeholder="Paste job description here...", label_visibility="collapsed")
        st.markdown("#### 📁 Upload Candidate Resumes")
        st.markdown("<small style='color:#555'>Upload multiple PDFs at once.</small>", unsafe_allow_html=True)
        rfs = st.file_uploader("RS", type=["pdf"], accept_multiple_files=True, key="em", label_visibility="collapsed")
        rb = st.button("🏆 Rank Candidates", use_container_width=True)

    with cr:
        if rb:
            if not ejd.strip(): st.warning("Please enter a job description.")
            elif not rfs: st.warning("Please upload at least one resume.")
            else:
                with st.spinner("Processing all resumes..."):
                    js = extract_skills(ejd)
                    if not js:
                        st.error("No skills found in job description.")
                    else:
                        st.markdown("#### 📋 Required Skills")
                        st.markdown(pills(js, "pill-neutral"), unsafe_allow_html=True)
                        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

                        results = []
                        for f in rfs:
                            rt = extract_text(f)
                            rs = extract_skills(rt)
                            matched, missing, ks = keyword_match(rs, js)
                            ss = sem_score(rt, ejd)
                            hs = hybrid(ks, ss)
                            results.append({"name": f.name.replace(".pdf","").replace("_"," ").replace("-"," ").title(), "ks": ks, "ss": ss, "hs": hs, "matched": matched, "missing": missing})

                        results.sort(key=lambda x: x["hs"], reverse=True)
                        st.markdown(f"#### 🏆 Candidate Rankings — {len(results)} Candidates")

                        for i, r in enumerate(results):
                            rank = i + 1
                            medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"#{rank}"
                            cc = color_cls(r["hs"])
                            with st.expander(f"{medal}  {r['name']}  —  Hybrid: {r['hs']}%"):
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown(f'<div style="text-align:center;padding:16px"><p style="color:#555;font-size:11px;text-transform:uppercase;letter-spacing:2px">Hybrid Score</p><div class="score-big {cc}" style="font-size:48px">{r["hs"]}%</div></div>', unsafe_allow_html=True)
                                    st.progress(int(min(r["hs"], 100)) / 100)
                                    st.markdown(f'<div class="score-mini-wrap" style="margin-top:8px"><div class="score-mini"><div class="slabel">Keyword</div><div class="svalue" style="color:#c8f04a;font-size:20px">{r["ks"]}%</div></div><div class="score-mini"><div class="slabel">Semantic</div><div class="svalue" style="color:#9fa8da;font-size:20px">{r["ss"]}%</div></div></div>', unsafe_allow_html=True)
                                with c2:
                                    st.markdown("**✅ Matched Skills**")
                                    st.markdown(pills(r["matched"], "pill-matched") if r["matched"] else "<span style='color:#666'>None</span>", unsafe_allow_html=True)
                                    st.markdown("**❌ Missing Skills**")
                                    st.markdown(pills(r["missing"], "pill-missing") if r["missing"] else "<span style='color:#4caf50'>All skills present!</span>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card" style="text-align:center;padding:48px"><div style="font-size:48px">🏢</div><p style="color:#555;margin-top:12px">Enter a job description and upload resumes, then click <strong style="color:#c8f04a">Rank Candidates</strong>.</p></div>', unsafe_allow_html=True)
