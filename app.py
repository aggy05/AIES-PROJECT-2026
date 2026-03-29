import streamlit as st
import pdfplumber
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ResuMatch – AI Hiring Platform", page_icon="🎯", layout="wide")

# ── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0d0d; color: #f0ece4; }

[data-testid="stSidebar"] { background: #0f0f0f; border-right: 1px solid #1e1e1e; }
[data-testid="stSidebar"] * { color: #f0ece4 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px !important; }

/* Buttons */
.stButton > button {
    background: #c8f04a; color: #0d0d0d; font-family: 'Syne', sans-serif;
    font-weight: 700; border: none; border-radius: 10px;
    padding: 12px 28px; font-size: 15px; width: 100%;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover { background: #d6f760; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: #141414 !important; color: #f0ece4 !important;
    border: 1px solid #2a2a2a !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 14px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus { border-color: #c8f04a !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #141414; border: 1.5px dashed #2a2a2a;
    border-radius: 10px; padding: 8px;
}
[data-testid="stFileUploader"]:hover { border-color: #c8f04a; }

/* Progress bar */
.stProgress > div > div > div > div { background: #c8f04a !important; border-radius: 4px; }

/* Expander */
.streamlit-expanderHeader { background: #141414 !important; border-radius: 8px !important; color: #f0ece4 !important; }
.streamlit-expanderContent { background: #0f0f0f !important; border: 1px solid #1e1e1e !important; }

/* Spinner */
.stSpinner > div { border-top-color: #c8f04a !important; }

/* Radio */
.stRadio label { color: #f0ece4 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── DATA ────────────────────────────────────────────────────────────────────
SKILLS_DB = {
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

# ── FUNCTIONS ───────────────────────────────────────────────────────────────
def extract_text(f):
    with pdfplumber.open(io.BytesIO(f.read())) as pdf:
        return "".join(p.extract_text() or "" for p in pdf.pages)

def extract_skills(text):
    tl = text.lower()
    return [s for s, syns in SKILLS_DB.items() if any(syn in tl for syn in syns)]

def keyword_match(rs, js):
    matched = [s for s in rs if s in js]
    missing = [s for s in js if s not in rs]
    score = round(len(matched) / len(js) * 100, 2) if js else 0
    return matched, missing, score

def sem_score(rt, jt):
    v = TfidfVectorizer(stop_words='english')
    vecs = v.fit_transform([jt, rt])
    return round(cosine_similarity(vecs[0], vecs[1])[0][0] * 100, 2)

def hybrid(k, s): return round(k * 0.6 + s * 0.4, 2)

def ai_feedback(rt, matched, missing, score):
    tl = rt.lower()
    fb = []
    if score >= 70:   fb.append(("✅", "Strong match", "Your resume aligns well with the job. Focus on polishing the presentation."))
    elif score >= 40: fb.append(("⚠️", "Moderate match", "Targeted improvements will significantly boost your chances."))
    else:             fb.append(("❌", "Low match", "Your resume needs significant work to align with this role."))

    wc = len(rt.split())
    if wc < 200:    fb.append(("📄", "Too short", "Aim for 400–600 words covering skills, experience, and projects."))
    elif wc > 1000: fb.append(("📄", "Too long", "Keep it concise — 1 page for freshers, 2 pages max for experienced candidates."))
    else:           fb.append(("📄", "Good length", f"Your resume has {wc} words — a solid length."))

    if any(c.isdigit() for c in rt):
        fb.append(("📊", "Metrics found", "Numbers and metrics strengthen your impact statements. Great work!"))
    else:
        fb.append(("📊", "No metrics", "Add quantifiable achievements like 'Improved accuracy by 20%' to stand out."))

    if "project" in tl:
        fb.append(("🔨", "Projects found", "Make sure each project states the problem, tools used, and outcome clearly."))
    else:
        fb.append(("🔨", "No projects", "Add 2–3 relevant projects — they significantly improve your profile."))

    if any(w in tl for w in ["bachelor","master","b.e","b.tech","mba","degree","university","college"]):
        fb.append(("🎓", "Education found", "Mention your CGPA if it's above 7.0 to strengthen your profile."))
    else:
        fb.append(("🎓", "Education unclear", "Ensure your degree, university, and graduation year are clearly visible."))

    if any(w in tl for w in ["certified","certification","certificate","coursera","udemy","nptel"]):
        fb.append(("🏅", "Certifications found", "These add strong credibility, especially for technical roles."))
    else:
        fb.append(("🏅", "No certifications", "Add relevant ones from Coursera, Udemy, or NPTEL to strengthen your profile."))

    return fb

def color_cls(s):
    return "#c8f04a" if s >= 70 else "#f0b429" if s >= 40 else "#f05a4a"

def pills(skills, bg, border, text):
    return "".join(
        f'<span style="display:inline-block;background:{bg};border:1px solid {border};color:{text};'
        f'border-radius:20px;padding:5px 14px;margin:4px;font-size:13px;font-weight:500">{s.title()}</span>'
        for s in skills
    )

def matched_pills(s): return pills(s, "#1a321a", "#4caf50", "#81c784")
def missing_pills(s): return pills(s, "#321a1a", "#f05a4a", "#ef9a9a")
def neutral_pills(s): return pills(s, "#1a1a2e", "#5c6bc0", "#9fa8da")

def score_bar(label, val, color):
    return f"""
    <div style="margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;font-size:12px;color:#888;margin-bottom:5px">
            <span>{label}</span><span style="color:{color};font-weight:500">{val}%</span>
        </div>
        <div style="background:#1e1e1e;border-radius:4px;height:8px;overflow:hidden">
            <div style="width:{min(val,100)}%;height:100%;background:{color};border-radius:4px;
                transition:width 0.8s ease"></div>
        </div>
    </div>"""

def donut_svg(score, color):
    circ = 2 * 3.14159 * 45
    offset = circ - (circ * score / 100)
    return f"""
    <svg viewBox="0 0 120 120" width="160" height="160">
        <circle cx="60" cy="60" r="45" fill="none" stroke="#1e1e1e" stroke-width="10"/>
        <circle cx="60" cy="60" r="45" fill="none" stroke="{color}" stroke-width="10"
            stroke-linecap="round" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}"
            transform="rotate(-90 60 60)"/>
        <text x="60" y="55" text-anchor="middle" font-family="Syne,sans-serif"
            font-size="20" font-weight="800" fill="{color}">{score}%</text>
        <text x="60" y="72" text-anchor="middle" font-family="DM Sans,sans-serif"
            font-size="9" fill="#555">hybrid score</text>
    </svg>"""

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px">
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#f0ece4">
            Resu<span style="color:#c8f04a">Match</span>
        </div>
        <div style="font-size:11px;color:#444;margin-top:2px;letter-spacing:1px;text-transform:uppercase">
            AI Hiring Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    portal = st.radio("", ["🏠  Home", "👤  Job Seeker Portal", "🏢  Employer Portal"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#333;line-height:1.8'>CSE (AIDS) · A3<br>Afiya · Agnes · Purvi<br>AIES Project 2026</div>", unsafe_allow_html=True)

# ── HOME ─────────────────────────────────────────────────────────────────────
if portal == "🏠  Home":
    import streamlit.components.v1 as components
    components.html("""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;color:#f0ece4;font-family:'DM Sans',sans-serif;padding:12px 8px 32px}
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes spin{to{stroke-dashoffset:0}}
.f1{animation:fadeUp 0.5s ease both}
.f2{animation:fadeUp 0.5s 0.12s ease both}
.f3{animation:fadeUp 0.5s 0.24s ease both}
.f4{animation:fadeUp 0.5s 0.36s ease both}
.f5{animation:fadeUp 0.5s 0.48s ease both}

.hero{text-align:center;padding:2rem 1rem 1.4rem}
.badge{display:inline-block;background:rgba(200,240,74,0.08);border:1px solid rgba(200,240,74,0.2);
  color:#8db83a;border-radius:20px;padding:5px 16px;font-size:11px;font-weight:500;
  letter-spacing:2px;text-transform:uppercase;margin-bottom:18px}
h1{font-family:'Syne',sans-serif;font-size:54px;font-weight:800;line-height:1.05;
  letter-spacing:-2px;color:#f0ece4;margin-bottom:10px}
h1 span{color:#c8f04a}
.sub{color:#555;font-size:15px;max-width:440px;margin:0 auto;line-height:1.6}

.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}
.stat{background:#111;border:1px solid #1e1e1e;border-radius:14px;padding:18px;text-align:center}
.stat-val{font-family:'Syne',sans-serif;font-size:38px;font-weight:800;color:#c8f04a;line-height:1}
.stat-lbl{font-size:10px;color:#444;margin-top:5px;text-transform:uppercase;letter-spacing:1px}

.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}
.card{background:#111;border:1px solid #1e1e1e;border-radius:16px;padding:22px}
.card.accent{border-color:rgba(200,240,74,0.4)}
.card-icon{font-size:26px;margin-bottom:10px}
.card h3{font-family:'Syne',sans-serif;font-size:16px;font-weight:800;margin-bottom:8px}
.card h3.green{color:#c8f04a}
.card p{color:#555;font-size:13px;line-height:1.55;margin-bottom:12px}
.feat{display:flex;align-items:center;gap:7px;font-size:12px;color:#555;margin-bottom:6px}
.feat span.tick{color:#c8f04a;font-weight:700}

.demo-section{margin-bottom:20px}
.section-label{font-size:10px;color:#444;text-transform:uppercase;letter-spacing:2px;
  font-weight:600;margin-bottom:12px}
.demo-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.demo-card{background:#111;border:1px solid #1e1e1e;border-radius:14px;padding:18px}
.demo-card h4{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;margin-bottom:14px;color:#f0ece4}

.donut-center{display:flex;flex-direction:column;align-items:center}
.score-sub{display:flex;gap:20px;margin-top:10px}
.score-mini{text-align:center}
.score-mini .v{font-family:'Syne',sans-serif;font-size:18px;font-weight:800;line-height:1}
.score-mini .l{font-size:10px;color:#444;text-transform:uppercase;letter-spacing:1px;margin-top:2px}

.bar-row{margin-bottom:10px}
.bar-top{display:flex;justify-content:space-between;font-size:11px;color:#666;margin-bottom:4px}
.bar-track{background:#1e1e1e;border-radius:4px;height:7px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}

.pills{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px}
.pill{border-radius:20px;padding:5px 13px;font-size:11px;font-weight:500}
.pill.m{background:rgba(76,175,80,0.12);border:1px solid rgba(76,175,80,0.3);color:#81c784}
.pill.x{background:rgba(240,90,74,0.1);border:1px solid rgba(240,90,74,0.25);color:#ef9a9a}

.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.step{background:#111;border:1px solid #1e1e1e;border-radius:14px;padding:16px;text-align:center}
.step-num{font-family:'Syne',sans-serif;font-size:10px;font-weight:800;color:#222;
  letter-spacing:2px;margin-bottom:6px}
.step-icon{font-size:22px;margin-bottom:6px}
.step-title{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;color:#f0ece4;margin-bottom:4px}
.step-desc{font-size:11px;color:#444;line-height:1.4}
</style>
</head>
<body>

<div class="hero f1">
  <div class="badge">AI · NLP · Semantic Matching</div>
  <h1>Resu<span>Match</span></h1>
  <p class="sub">AI-powered resume & job description matching — smarter than keywords alone</p>
</div>

<div class="stats f2">
  <div class="stat"><div class="stat-val" id="s1">0</div><div class="stat-lbl">Skills tracked</div></div>
  <div class="stat"><div class="stat-val" id="s2">0</div><div class="stat-lbl">Match engines</div></div>
  <div class="stat"><div class="stat-val" id="s3">0</div><div class="stat-lbl">AI feedback checks</div></div>
</div>

<div class="cards f3">
  <div class="card accent">
    <div class="card-icon">👤</div>
    <h3 class="green">Job Seeker Portal</h3>
    <p>Upload your resume + paste a job description to get a full AI analysis with match score, skill gaps, and tips.</p>
    <div class="feat"><span class="tick">✓</span> Hybrid keyword + semantic score</div>
    <div class="feat"><span class="tick">✓</span> Matched & missing skill breakdown</div>
    <div class="feat"><span class="tick">✓</span> AI-powered resume feedback</div>
    <div class="feat"><span class="tick">✓</span> Personalised learning resources</div>
  </div>
  <div class="card">
    <div class="card-icon">🏢</div>
    <h3>Employer Portal</h3>
    <p>Post a job + upload multiple resumes. Candidates ranked automatically using a hybrid AI scoring engine.</p>
    <div class="feat"><span class="tick">✓</span> Multi-resume batch processing</div>
    <div class="feat"><span class="tick">✓</span> Auto candidate ranking</div>
    <div class="feat"><span class="tick">✓</span> Keyword + semantic hybrid score</div>
    <div class="feat"><span class="tick">✓</span> Per-candidate skill analysis</div>
  </div>
</div>

<div class="demo-section f4">
  <div class="section-label">Live demo — sample analysis</div>
  <div class="demo-grid">
    <div class="demo-card">
      <h4>Match score</h4>
      <div class="donut-center">
        <svg width="140" height="140" viewBox="0 0 140 140">
          <circle cx="70" cy="70" r="52" fill="none" stroke="#1e1e1e" stroke-width="11"/>
          <circle id="ring" cx="70" cy="70" r="52" fill="none" stroke="#c8f04a" stroke-width="11"
            stroke-linecap="round" stroke-dasharray="326.7" stroke-dashoffset="326.7"
            transform="rotate(-90 70 70)" style="transition:stroke-dashoffset 1.4s cubic-bezier(.4,0,.2,1)"/>
          <text x="70" y="63" text-anchor="middle" font-family="Syne,sans-serif"
            font-size="24" font-weight="800" fill="#c8f04a" id="scoreText">0%</text>
          <text x="70" y="80" text-anchor="middle" font-family="DM Sans,sans-serif"
            font-size="10" fill="#555">hybrid score</text>
        </svg>
        <div class="score-sub">
          <div class="score-mini"><div class="v" style="color:#9fa8da" id="kscore">0%</div><div class="l">keyword</div></div>
          <div class="score-mini"><div class="v" style="color:#c8f04a" id="sscore">0%</div><div class="l">semantic</div></div>
        </div>
      </div>
    </div>

    <div class="demo-card">
      <h4>Skill match breakdown</h4>
      <div class="bar-row"><div class="bar-top"><span>Python</span><span style="color:#c8f04a">100%</span></div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#c8f04a;transition:width 1s 0.2s ease" data-w="100"></div></div></div>
      <div class="bar-row"><div class="bar-top"><span>SQL</span><span style="color:#c8f04a">100%</span></div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#c8f04a;transition:width 1s 0.35s ease" data-w="100"></div></div></div>
      <div class="bar-row"><div class="bar-top"><span>Data Analysis</span><span style="color:#c8f04a">100%</span></div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#c8f04a;transition:width 1s 0.5s ease" data-w="100"></div></div></div>
      <div class="bar-row"><div class="bar-top"><span>Machine Learning</span><span style="color:#f0b429">80%</span></div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#f0b429;transition:width 1s 0.65s ease" data-w="80"></div></div></div>
      <div class="bar-row"><div class="bar-top"><span>TensorFlow</span><span style="color:#f05a4a">0%</span></div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#f05a4a;transition:width 1s 0.8s ease" data-w="0"></div></div></div>
      <div class="bar-row"><div class="bar-top"><span>Communication</span><span style="color:#f05a4a">0%</span></div><div class="bar-track"><div class="bar-fill" style="width:0%;background:#f05a4a;transition:width 1s 0.95s ease" data-w="0"></div></div></div>
    </div>
  </div>

  <div style="background:#111;border:1px solid #1e1e1e;border-radius:14px;padding:16px;margin-top:12px">
    <div class="section-label" style="margin-bottom:10px">Matched & missing skills</div>
    <div class="pills">
      <span class="pill m">Python</span><span class="pill m">SQL</span>
      <span class="pill m">Data Analysis</span><span class="pill m">Deep Learning</span>
      <span class="pill m">NLP</span><span class="pill m">Pandas</span>
      <span class="pill x">TensorFlow</span><span class="pill x">Communication</span>
      <span class="pill x">Problem Solving</span>
    </div>
  </div>
</div>

<div class="f5">
  <div class="section-label" style="margin-bottom:12px">How it works</div>
  <div class="steps">
    <div class="step"><div class="step-num">01</div><div class="step-icon">📄</div><div class="step-title">Parse Resume</div><div class="step-desc">PDF text extracted automatically</div></div>
    <div class="step"><div class="step-num">02</div><div class="step-icon">🔍</div><div class="step-title">Extract Skills</div><div class="step-desc">Keyword + synonym detection</div></div>
    <div class="step"><div class="step-num">03</div><div class="step-icon">🧠</div><div class="step-title">Semantic Match</div><div class="step-desc">TF-IDF cosine similarity</div></div>
    <div class="step"><div class="step-num">04</div><div class="step-icon">🎯</div><div class="step-title">AI Feedback</div><div class="step-desc">Gap analysis + improvement tips</div></div>
  </div>
</div>

<script>
setTimeout(() => {
  // Animate counters
  [{id:'s1',target:24},{id:'s2',target:3},{id:'s3',target:7}].forEach(({id,target}) => {
    let v=0; const s=target/40;
    const t=setInterval(()=>{v=Math.min(v+s,target);document.getElementById(id).textContent=Math.round(v);if(v>=target)clearInterval(t);},30);
  });
  // Animate donut
  const score=72, circ=326.7;
  document.getElementById('ring').style.strokeDashoffset = circ-(circ*score/100);
  let n=0; const dt=setInterval(()=>{n=Math.min(n+2,score);document.getElementById('scoreText').textContent=n+'%';if(n>=score)clearInterval(dt);},20);
  // Keyword / semantic labels
  setTimeout(()=>{document.getElementById('kscore').textContent='60%';document.getElementById('sscore').textContent='10%';},500);
  // Animate bars
  document.querySelectorAll('.bar-fill').forEach(b=>{setTimeout(()=>{b.style.width=b.dataset.w+'%';},100);});
},400);
</script>
</body>
</html>
""", height=1100, scrolling=False)

# ── JOB SEEKER PORTAL ────────────────────────────────────────────────────────
elif portal == "👤  Job Seeker Portal":
    st.markdown("""
    <div style="margin-bottom:24px">
        <h2 style="font-family:'Syne',sans-serif;font-size:30px;font-weight:800;color:#f0ece4;margin-bottom:4px">
            👤 Job Seeker Portal
        </h2>
        <p style="color:#444;font-size:14px">Upload your resume + paste a job description to get your full AI match report.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT ROW (side by side) ──
    cl, cr = st.columns([1, 1], gap="large")
    with cl:
        st.markdown("<p style='color:#888;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>📄 Upload Resume (PDF)</p>", unsafe_allow_html=True)
        rf = st.file_uploader("r", type=["pdf"], key="sk", label_visibility="collapsed")
    with cr:
        st.markdown("<p style='color:#888;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>📋 Job Description</p>", unsafe_allow_html=True)
        jd = st.text_area("j", height=160, placeholder="Paste the job description here...", label_visibility="collapsed")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    go = st.button("🔍  Analyse My Resume", use_container_width=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── RESULTS (full width below) ──
    if go:
        if not rf:
            st.warning("Please upload your resume PDF.")
        elif not jd.strip():
            st.warning("Please paste a job description.")
        else:
            with st.spinner("Analysing your resume..."):
                rt = extract_text(rf)
                rs = extract_skills(rt)
                js = extract_skills(jd)

            if not js:
                st.error("No recognisable skills found in the job description.")
            else:
                matched, missing, ks = keyword_match(rs, js)
                ss = sem_score(rt, jd)
                hs = hybrid(ks, ss)
                hcol = color_cls(hs)

                # ── ROW 1: Score card + skill pills ──
                r1c1, r1c2 = st.columns([1, 2], gap="large")

                with r1c1:
                    st.markdown(f"""
                    <div style="background:#111;border:1px solid #1e1e1e;border-radius:16px;
                        padding:28px;text-align:center">
                        <div style="font-size:11px;color:#444;text-transform:uppercase;
                            letter-spacing:2px;margin-bottom:16px">Hybrid Match Score</div>
                        {donut_svg(hs, hcol)}
                        <div style="margin-top:20px">
                            {score_bar("Keyword", ks, "#9fa8da")}
                            {score_bar("Semantic", ss, "#c8f04a")}
                            {score_bar("Hybrid", hs, hcol)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with r1c2:
                    st.markdown(f"""
                    <div style="background:#111;border:1px solid #1e1e1e;border-radius:16px;
                        padding:24px;margin-bottom:14px">
                        <div style="font-size:11px;color:#444;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:12px">✅ Matched Skills ({len(matched)})</div>
                        {matched_pills(matched) if matched else '<span style="color:#333;font-size:13px">None found</span>'}
                    </div>
                    <div style="background:#111;border:1px solid #1e1e1e;border-radius:16px;padding:24px">
                        <div style="font-size:11px;color:#444;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:12px">❌ Missing Skills ({len(missing)})</div>
                        {missing_pills(missing) if missing else '<span style="color:#4caf50;font-size:14px">🎉 All required skills present!</span>'}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                # ── ROW 2: AI Feedback + Suggestions side by side ──
                r2c1, r2c2 = st.columns([1, 1], gap="large")

                with r2c1:
                    st.markdown("<div style='font-size:11px;color:#444;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>🤖 AI Resume Feedback</div>", unsafe_allow_html=True)
                    fb = ai_feedback(rt, matched, missing, hs)
                    fb_html = ""
                    for icon, title, desc in fb:
                        fb_html += f"""
                        <div style="background:#111;border:1px solid #1e1e1e;border-radius:12px;
                            padding:14px 16px;margin-bottom:8px;display:flex;gap:12px;align-items:flex-start">
                            <span style="font-size:18px;flex-shrink:0;margin-top:1px">{icon}</span>
                            <div>
                                <div style="font-family:'Syne',sans-serif;font-size:12px;font-weight:700;
                                    color:#c8f04a;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">{title}</div>
                                <div style="font-size:13px;color:#666;line-height:1.5">{desc}</div>
                            </div>
                        </div>"""
                    st.markdown(fb_html, unsafe_allow_html=True)

                with r2c2:
                    if missing:
                        st.markdown("<div style='font-size:11px;color:#444;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>💡 How to Fill the Gaps</div>", unsafe_allow_html=True)
                        sug_html = ""
                        for sk in missing:
                            tip = SUGGESTIONS.get(sk, f"Consider learning {sk.title()} through Coursera or Udemy.")
                            sug_html += f"""
                            <div style="background:#111;border:1px solid #1e1e1e;border-radius:12px;
                                padding:14px 16px;margin-bottom:8px">
                                <div style="font-size:11px;color:#9fa8da;font-weight:700;text-transform:uppercase;
                                    letter-spacing:1px;margin-bottom:5px">{sk.upper()}</div>
                                <div style="font-size:13px;color:#555;line-height:1.5">{tip}</div>
                            </div>"""
                        st.markdown(sug_html, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background:#111;border:1px solid #1e1e1e;border-radius:16px;
                            padding:40px;text-align:center">
                            <div style="font-size:36px;margin-bottom:12px">🎉</div>
                            <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
                                color:#c8f04a">Perfect match!</div>
                            <div style="font-size:13px;color:#444;margin-top:6px">
                                Your resume covers all required skills.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                with st.expander("📋 All skills detected in your resume"):
                    st.markdown(neutral_pills(rs) if rs else "<span style='color:#333'>None found</span>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#111;border:1px solid #1e1e1e;border-radius:16px;
            padding:60px 40px;text-align:center">
            <div style="font-size:48px;margin-bottom:16px">🎯</div>
            <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                color:#f0ece4;margin-bottom:8px">Ready to analyse</div>
            <div style="font-size:14px;color:#444;line-height:1.6;max-width:300px;margin:0 auto">
                Upload your resume and paste a job description, then click
                <span style="color:#c8f04a;font-weight:500">Analyse My Resume</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── EMPLOYER PORTAL ──────────────────────────────────────────────────────────
elif portal == "🏢  Employer Portal":
    st.markdown("""
    <div style="margin-bottom:24px">
        <h2 style="font-family:'Syne',sans-serif;font-size:30px;font-weight:800;color:#f0ece4;margin-bottom:4px">
            🏢 Employer Portal
        </h2>
        <p style="color:#444;font-size:14px">Post a job + upload multiple resumes — candidates ranked by hybrid AI score.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT ROW (side by side) ──
    cl, cr = st.columns([1, 1], gap="large")
    with cl:
        st.markdown("<p style='color:#888;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>📋 Job Description</p>", unsafe_allow_html=True)
        ejd = st.text_area("ej", height=180, placeholder="Paste the job description here...", label_visibility="collapsed")
    with cr:
        st.markdown("<p style='color:#888;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>📁 Candidate Resumes</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#333;font-size:12px;margin-bottom:8px'>Upload multiple PDFs at once</p>", unsafe_allow_html=True)
        rfs = st.file_uploader("rs", type=["pdf"], accept_multiple_files=True, key="em", label_visibility="collapsed")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    rb = st.button("🏆  Rank Candidates", use_container_width=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── RESULTS (full width below) ──
    if rb:
        if not ejd.strip():
            st.warning("Please enter a job description.")
        elif not rfs:
            st.warning("Please upload at least one resume.")
        else:
            with st.spinner("Processing all resumes..."):
                js = extract_skills(ejd)
                if not js:
                    st.error("No recognisable skills found in the job description.")
                else:
                    results = []
                    for f in rfs:
                        rt = extract_text(f)
                        rs = extract_skills(rt)
                        matched, missing, ks = keyword_match(rs, js)
                        ss = sem_score(rt, ejd)
                        hs = hybrid(ks, ss)
                        results.append({
                            "name": f.name.replace(".pdf","").replace("_"," ").replace("-"," ").title(),
                            "ks": ks, "ss": ss, "hs": hs,
                            "matched": matched, "missing": missing
                        })
                    results.sort(key=lambda x: x["hs"], reverse=True)

                    # Required skills + summary row
                    rc1, rc2, rc3 = st.columns(3, gap="medium")
                    with rc1:
                        st.markdown(f"""
                        <div style="background:#111;border:1px solid #1e1e1e;border-radius:14px;
                            padding:18px;text-align:center">
                            <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;
                                color:#c8f04a">{len(results)}</div>
                            <div style="font-size:11px;color:#444;text-transform:uppercase;
                                letter-spacing:1px;margin-top:4px">Candidates</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with rc2:
                        st.markdown(f"""
                        <div style="background:#111;border:1px solid #1e1e1e;border-radius:14px;
                            padding:18px;text-align:center">
                            <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;
                                color:#c8f04a">{len(js)}</div>
                            <div style="font-size:11px;color:#444;text-transform:uppercase;
                                letter-spacing:1px;margin-top:4px">Required Skills</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with rc3:
                        top_score = results[0]["hs"] if results else 0
                        st.markdown(f"""
                        <div style="background:#111;border:1px solid #1e1e1e;border-radius:14px;
                            padding:18px;text-align:center">
                            <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;
                                color:#c8f04a">{top_score}%</div>
                            <div style="font-size:11px;color:#444;text-transform:uppercase;
                                letter-spacing:1px;margin-top:4px">Top Score</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                    # Required skills pills
                    st.markdown(f"""
                    <div style="background:#111;border:1px solid #1e1e1e;border-radius:14px;
                        padding:18px;margin-bottom:20px">
                        <div style="font-size:11px;color:#444;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:12px">Required Skills for This Role</div>
                        {neutral_pills(js)}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"<div style='font-size:11px;color:#444;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px'>🏆 Candidate Rankings</div>", unsafe_allow_html=True)

                    medals = ["🥇", "🥈", "🥉"]
                    for i, r in enumerate(results):
                        medal = medals[i] if i < 3 else f"#{i+1}"
                        hcol = color_cls(r["hs"])
                        with st.expander(f"{medal}  {r['name']}  ·  Hybrid Score: {r['hs']}%"):
                            ec1, ec2, ec3 = st.columns([1, 1, 1], gap="medium")
                            with ec1:
                                st.markdown(f"""
                                <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:14px;
                                    padding:20px;text-align:center">
                                    <div style="font-size:11px;color:#444;text-transform:uppercase;
                                        letter-spacing:1px;margin-bottom:12px">Match Score</div>
                                    {donut_svg(r['hs'], hcol)}
                                    <div style="margin-top:16px">
                                        {score_bar("Keyword", r['ks'], "#9fa8da")}
                                        {score_bar("Semantic", r['ss'], "#c8f04a")}
                                        {score_bar("Hybrid", r['hs'], hcol)}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            with ec2:
                                st.markdown(f"""
                                <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:14px;
                                    padding:20px;height:100%">
                                    <div style="font-size:11px;color:#444;text-transform:uppercase;
                                        letter-spacing:1px;margin-bottom:12px">✅ Matched Skills ({len(r['matched'])})</div>
                                    {matched_pills(r['matched']) if r['matched'] else '<span style="color:#333;font-size:13px">None found</span>'}
                                </div>
                                """, unsafe_allow_html=True)
                            with ec3:
                                st.markdown(f"""
                                <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:14px;
                                    padding:20px;height:100%">
                                    <div style="font-size:11px;color:#444;text-transform:uppercase;
                                        letter-spacing:1px;margin-bottom:12px">❌ Missing Skills ({len(r['missing'])})</div>
                                    {missing_pills(r['missing']) if r['missing'] else '<span style="color:#4caf50;font-size:13px">All present!</span>'}
                                </div>
                                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#111;border:1px solid #1e1e1e;border-radius:16px;
            padding:60px 40px;text-align:center">
            <div style="font-size:48px;margin-bottom:16px">🏢</div>
            <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                color:#f0ece4;margin-bottom:8px">Ready to rank</div>
            <div style="font-size:14px;color:#444;line-height:1.6;max-width:300px;margin:0 auto">
                Enter a job description and upload candidate resumes, then click
                <span style="color:#c8f04a;font-weight:500">Rank Candidates</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


