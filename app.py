"""
app.py — SmartATS: Professional Explainable Resume Ranker
Run with: streamlit run app.py
"""
import sys
import os
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
from parsers.parser import extract_text, extract_candidate_name, extract_contact_info, redact_pii
from engine.scorer import rank_resumes, route_resumes_to_open_jobs
from engine.jobs import save_job_opening, load_all_job_openings, delete_job_opening
from evaluation import load_hackathon_candidate_pool
from engine.llm_judge import (
    is_ollama_available,
    get_ollama_models,
    judge_with_ollama,
    judge_with_gemini,
    judge_with_simulation,
)


def calculate_gemini_embedding_similarity(text_a: str, text_b: str, api_key: str) -> float:
    import requests
    import math
    
    def get_embedding(text):
        url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
        if api_key.startswith("ya29."):
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        else:
            url = f"{url}?key={api_key}"
            headers = {"Content-Type": "application/json"}
        payload = {
            "model": "models/text-embedding-004",
            "content": {
                "parts": [{"text": text[:8000]}]  # Cap to prevent API errors on large resumes
            }
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()["embedding"]["values"]
        else:
            raise Exception(f"Gemini Embedding failed: {resp.text}")
            
    try:
        v1 = get_embedding(text_a)
        v2 = get_embedding(text_b)
        
        dot = sum(a*b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a*a for a in v1))
        norm_b = math.sqrt(sum(b*b for b in v2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = dot / (norm_a * norm_b)
        return float(similarity * 100.0)
    except Exception as e:
        # Fail gracefully to allow pipeline to continue
        raise e

def send_and_log_email(to_email: str, subject: str, body: str, smtp_settings: dict, candidate_name: str) -> dict:
    import datetime
    from engine.outreach import send_email
    res = send_email(to_email, subject, body, smtp_config=smtp_settings)
    
    log_entry = {
        "Candidate Name": candidate_name,
        "Recipient Email": to_email,
        "Subject": subject,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Status": "✅ Success" if res["success"] else "❌ Failed",
        "Mode": res.get("mode", "Simulated"),
        "Details": res.get("info") if res["success"] else res.get("error", "Unknown error")
    }
    
    if "outreach_log" not in st.session_state:
        st.session_state["outreach_log"] = []
    st.session_state["outreach_log"].append(log_entry)
    return res

# ---------- Page Config & Styling ----------
st.set_page_config(
    page_title="SmartATS — Explainable Hybrid Resume Ranker",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# Premium UI CSS Injection
import base64

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

banner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cyberpunk_nature_banner.png")
banner_b64 = get_base64_image(banner_path)

bg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cyberpunk_nature_background.png")
bg_b64 = get_base64_image(bg_path)

# Premium UI CSS Injection
css_content = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Fonts */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Hide Streamlit elements */
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    footer {
        visibility: hidden !important;
        height: 0 !important;
        position: absolute !important;
    }
    
    /* App background - Transparent to overlay on premium nature background */
    .stApp {
        background: transparent !important;
        color: #F3F4F6 !important;
    }
    
    /* Shifting organic canopy background */
    .nature-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -2;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(12, 8, 8, 0.75) 0%, rgba(8, 4, 4, 0.9) 100%),
                    url('data:image/png;base64,BACKGROUND_IMAGE_B64') !important;
        background-size: cover !important;
        background-position: center !important;
    }
    
    .nature-bg::before {
        content: '';
        position: absolute;
        top: -10%; left: -10%; right: -10%; bottom: -10%;
        background: radial-gradient(circle at 20% 35%, rgba(220, 38, 38, 0.22) 0%, transparent 60%),
                    radial-gradient(circle at 80% 65%, rgba(245, 158, 11, 0.18) 0%, transparent 60%);
        filter: blur(50px);
        animation: canopy-drift 24s ease-in-out infinite alternate;
    }
    
    @keyframes canopy-drift {
        0% { transform: scale(1) translate(0, 0); }
        100% { transform: scale(1.1) translate(3%, 2%); }
    }
    
    /* Live nature fireflies */
    .firefly {
        position: absolute;
        border-radius: 50%;
        background: rgba(245, 158, 11, 0.55);
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.8), 0 0 15px rgba(220, 38, 38, 0.4);
        animation: float-upward 25s infinite linear;
        opacity: 0;
    }
    
    .firefly:nth-child(1) { width: 5px; height: 5px; left: 12%; top: 90%; animation-duration: 22s; animation-delay: 0s; }
    .firefly:nth-child(2) { width: 7px; height: 7px; left: 35%; top: 95%; animation-duration: 28s; animation-delay: 2s; }
    .firefly:nth-child(3) { width: 4px; height: 4px; left: 52%; top: 92%; animation-duration: 19s; animation-delay: 4s; }
    .firefly:nth-child(4) { width: 8px; height: 8px; left: 68%; top: 97%; animation-duration: 31s; animation-delay: 1s; }
    .firefly:nth-child(5) { width: 6px; height: 6px; left: 88%; top: 91%; animation-duration: 24s; animation-delay: 5s; }
    .firefly:nth-child(6) { width: 5px; height: 5px; left: 22%; top: 94%; animation-duration: 23s; animation-delay: 3s; }
    .firefly:nth-child(7) { width: 7px; height: 7px; left: 45%; top: 98%; animation-duration: 29s; animation-delay: 6s; }
    .firefly:nth-child(8) { width: 4px; height: 4px; left: 61%; top: 93%; animation-duration: 18s; animation-delay: 2s; }
    .firefly:nth-child(9) { width: 9px; height: 9px; left: 78%; top: 96%; animation-duration: 34s; animation-delay: 0s; }
    .firefly:nth-child(10) { width: 6px; height: 6px; left: 93%; top: 92%; animation-duration: 26s; animation-delay: 4s; }

    @keyframes float-upward {
        0% { transform: translateY(0) translateX(0) scale(0.8); opacity: 0; }
        10% { opacity: 0.6; }
        90% { opacity: 0.6; }
        100% { transform: translateY(-90vh) translateX(50px) scale(1.2); opacity: 0; }
    }
    
    /* Translucent Header */
    header[data-testid="stHeader"] {
        background-color: rgba(12, 8, 8, 0.45) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid rgba(220, 38, 38, 0.15) !important;
    }
    
    /* Text Inputs, Selectboxes, Textareas - Glassmorphic Dark Style */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] input, 
    div[data-baseweb="textarea"] textarea,
    .stTextArea textarea, 
    .stTextInput input, 
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(30, 16, 16, 0.45) !important;
        color: #F3F4F6 !important;
        border: 1px solid rgba(220, 38, 38, 0.25) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.05) !important;
        transition: all 0.2s ease !important;
    }
    
    /* Input Hover/Focus States */
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] input:focus,
    div[data-baseweb="textarea"] textarea:focus,
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #FBBF24 !important;
        box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.25) !important;
        background-color: rgba(45, 20, 20, 0.6) !important;
    }
    
    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(220, 38, 38, 0.2);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(220, 38, 38, 0.45);
    }
    
    /* Sidebar glassmorphic styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 8, 8, 0.75) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(220, 38, 38, 0.15) !important;
    }
    
    section[data-testid="stSidebar"] [class*="css"] {
        color: #F3F4F6 !important;
    }
    
    /* Titles and Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', -apple-system, sans-serif;
        font-weight: 700;
        color: #F8FAFC !important;
        letter-spacing: -0.02em;
    }
    
    /* Cyber Glass Header Banner */
    .header-banner {
        background: linear-gradient(135deg, rgba(12, 8, 8, 0.8) 0%, rgba(12, 8, 8, 0.95) 100%),
                    url('data:image/png;base64,BANNER_IMAGE_B64') !important;
        background-size: cover !important;
        background-position: center !important;
        backdrop-filter: blur(30px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(30px) saturate(180%) !important;
        border: 1px solid rgba(220, 38, 38, 0.25) !important;
        padding: 3rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
    }
    
    .header-banner h1 {
        background: linear-gradient(90deg, #F8FAFC 0%, #F59E0B 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 0;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.025em;
    }
    .header-banner p {
        color: #E2E8F0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8) !important;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
        text-shadow: none !important;
    }
    
    /* File Uploader glass styling */
    div[data-testid="stFileUploader"] {
        background-color: rgba(30, 16, 16, 0.45) !important;
        border: 1px solid rgba(220, 38, 38, 0.25) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #FBBF24 !important;
        background-color: rgba(45, 20, 20, 0.6) !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: rgba(220, 38, 38, 0.15) !important;
        border: 1px solid rgba(220, 38, 38, 0.3) !important;
        color: #FBBF24 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #F59E0B !important;
        color: #0C0808 !important;
        border-color: #F59E0B !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 16, 16, 0.45) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(220, 38, 38, 0.25) !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .metric-card:hover {
        transform: translateY(-2px) !important;
        background: rgba(45, 20, 20, 0.6) !important;
        box-shadow: 0 8px 30px rgba(220, 38, 38, 0.15) !important;
    }
    .metric-label {
        font-size: 0.725rem !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
    }
    .metric-value {
        font-size: 1.6rem !important;
        color: #F8FAFC !important;
        font-weight: 700 !important;
        margin-top: 0.25rem !important;
    }
    
    /* Candidate Cards */
    .candidate-card {
        background: rgba(30, 16, 16, 0.45) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(220, 38, 38, 0.25) !important;
        border-left: 5px solid #F59E0B !important;
        border-radius: 16px !important;
        padding: 1.3rem 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .candidate-card.top-rank {
        border-left-color: #34D399 !important;
    }
    .candidate-card:hover {
        transform: translateY(-2px) scale(1.002) !important;
        background: rgba(45, 20, 20, 0.6) !important;
        box-shadow: 0 15px 40px rgba(220, 38, 38, 0.22), 0 0 20px rgba(245, 158, 11, 0.15) !important;
    }
    
    /* Expanders & Accordions Styling */
    .streamlit-expanderHeader {
        background-color: rgba(30, 16, 16, 0.35) !important;
        border: 1px solid rgba(220, 38, 38, 0.18) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.15s ease !important;
    }
    .streamlit-expanderHeader p {
        color: #F8FAFC !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: rgba(45, 20, 20, 0.5) !important;
        border-color: rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Primary button styling */
    div.stButton > button {
        background: linear-gradient(90deg, #DC2626 0%, #F59E0B 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #B91C1C 0%, #D97706 100%) !important;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Tabs custom styling - segment controller */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 16, 16, 0.35) !important;
        border-radius: 10px !important;
        padding: 0.2rem !important;
        border: 1px solid rgba(220, 38, 38, 0.2) !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: #DC2626 !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem !important;
        border-radius: 9999px !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        margin-right: 0.4rem !important;
        margin-bottom: 0.4rem !important;
        letter-spacing: 0.02em;
    }
    .badge-primary {
        background-color: rgba(220, 38, 38, 0.15) !important;
        color: #F87171 !important;
        border: 1px solid rgba(220, 38, 38, 0.25) !important;
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15) !important;
        color: #34D399 !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15) !important;
        color: #F87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.25) !important;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15) !important;
        color: #FBBF24 !important;
        border: 1px solid rgba(245, 158, 11, 0.25) !important;
    }
    
    /* LLM judge card - Glassmorphic Box */
    .llm-judge-box {
        background: rgba(30, 16, 16, 0.5) !important;
        border: 1px solid rgba(220, 38, 38, 0.18) !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        margin-top: 1rem;
    }
    </style>
    
    <div class="nature-bg">
        <div class="firefly"></div>
        <div class="firefly"></div>
        <div class="firefly"></div>
        <div class="firefly"></div>
        <div class="firefly"></div>
        <div class="firefly"></div>
        <div class="firefly"></div>
        <div class="firefly"></div>
        <div class="firefly"></div>
        <div class="firefly"></div>
    </div>
"""

st.markdown(css_content.replace("BANNER_IMAGE_B64", banner_b64).replace("BACKGROUND_IMAGE_B64", bg_b64), unsafe_allow_html=True)

# ---------- Helper Utilities ----------
def get_jd_presets() -> dict:
    """Loads default preset JD files from the workspace directory."""
    presets = {
        "Custom (Paste New)": ""
    }
    # Look for files in data/sample_jd/
    jd_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_jd")
    if os.path.exists(jd_dir):
        for f in sorted(os.listdir(jd_dir)):
            if f.endswith(".txt"):
                raw_name = f.replace(".txt", "")
                if raw_name == "backend_python_developer":
                    name = "Backend Python Developer"
                elif raw_name == "frontend_react_developer":
                    name = "Frontend React Developer"
                elif raw_name == "devops_sre_engineer":
                    name = "DevOps SRE Engineer"
                elif raw_name == "data_scientist_ml":
                    name = "Data Scientist (Machine Learning)"
                elif raw_name == "product_manager":
                    name = "Product Manager"
                elif raw_name == "security_engineer":
                    name = "Cybersecurity Analyst / Security Engineer"
                elif raw_name == "mobile_developer":
                    name = "Mobile iOS/Android Developer"
                else:
                    name = raw_name.replace("_", " ").title()
                try:
                    with open(os.path.join(jd_dir, f), "r") as file:
                        presets[name] = file.read()
                except Exception:
                    pass
    return presets


def get_sample_resumes() -> list:
    """Loads default preset resume files."""
    resumes = []
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_resumes")
    if os.path.exists(res_dir):
        for f in os.listdir(res_dir):
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(res_dir, f), "r") as file:
                        content = file.read()
                        resumes.append({
                            "filename": f,
                            "text": content
                        })
                except Exception:
                    pass
    return resumes


def extract_graduation_year(text: str) -> Optional[int]:
    """Helper to guess candidate graduation year from resume content (for bias audit)."""
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20[0-2]\d)\b", text)]
    import datetime; edu_years = [y for y in years if 1975 <= y <= datetime.datetime.now().year + 2]
    if edu_years:
        return max(edu_years)
    return None


# ---------- Sidebar: Core Settings ----------
st.sidebar.title("🎯 SmartATS settings")
st.sidebar.caption("Enterprise Hybrid Resume Scorer")
st.sidebar.markdown("---")

# 1. Custom Weights Configurator
with st.sidebar.expander("🛠️ Interactive Custom Weights", expanded=True):
    st.caption("Change weights of the 4 scoring dimensions. Total must equal 100%.")
    w_skills = st.slider("🛠️ Skill Match Weight (%)", 0, 100, 40)
    w_exp = st.slider("💼 Experience Weight (%)", 0, 100, 25)
    w_edu = st.slider("🎓 Education Weight (%)", 0, 100, 15)
    w_sem = st.slider("🧠 Semantic Similarity (%)", 0, 100, 20)

    total_w = w_skills + w_exp + w_edu + w_sem
    if total_w == 100:
        st.success(f"Total: {total_w}% (Balanced)")
    else:
        st.warning(f"Total: {total_w}% (Auto-normalizing to 100%)")

    weights_dict = {
        "skills": w_skills / 100.0,
        "experience": w_exp / 100.0,
        "education": w_edu / 100.0,
        "semantic": w_sem / 100.0,
    }
    if total_w != 100 and total_w > 0:
        for k in weights_dict:
            weights_dict[k] = weights_dict[k] / (total_w / 100.0)

st.sidebar.markdown("---")

# 2. Multi-Engine LLM Judge Configurator
st.sidebar.subheader("🧠 LLM Qualitative Judge")
llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Simulated LLM (Offline)", "Google Gemini (Cloud)", "Ollama (Local)", "Disabled"],
    index=0,
    help="Select how qualitative reviews and bounded scoring adjustments are evaluated."
)

llm_fn = None
ollama_url = "http://localhost:11434"
ollama_model = "llama3.2:1b"
gemini_model = "gemini-2.0-flash"
gemini_key = ""

if llm_provider == "Ollama (Local)":
    ollama_url = st.sidebar.text_input("Ollama Endpoint URL", value=ollama_url)
    available_models = get_ollama_models(ollama_url)
    if available_models:
        ollama_model = st.sidebar.selectbox("Ollama Model", available_models)
    else:
        st.sidebar.warning("⚠️ Could not connect to local Ollama. Verify it is running (`ollama serve`).")
        ollama_model = st.sidebar.text_input("Specify Model Name", value=ollama_model)
    st.sidebar.warning("⚠️ **Performance Note:** Running models locally via Ollama requires significant computing power. Screening multiple candidates can be slow depending on your hardware. For faster screening, consider using Google Gemini or the Simulated LLM.")

    # Test connection
    if st.sidebar.button("🔌 Test Ollama Connection", use_container_width=True):
        if is_ollama_available(ollama_url):
            st.sidebar.success("🟢 Connection successful!")
        else:
            st.sidebar.error("🔴 Connection failed. Ollama not running.")

    llm_fn = lambda r_text, jd, algo_res, _url=ollama_url, _model=ollama_model: judge_with_ollama(r_text, jd, algo_res, ollama_url=_url, model=_model)

elif llm_provider == "Google Gemini (Cloud)":
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Provide a Gemini API Key to connect to Google Cloud."
    )
    gemini_model = st.sidebar.selectbox("Gemini Model", ["gemini-2.0-flash", "gemini-2.5-flash-preview", "gemini-2.5-pro-preview", "gemini-1.5-flash"])

    if st.sidebar.button("🔌 Test Gemini Connection", use_container_width=True):
        if not gemini_key:
            st.sidebar.error("Gemini API key is required.")
        else:
            try:
                url = f"https://generativelanguage.googleapis.com/v1/models/{gemini_model}:generateContent"
                headers = {"Content-Type": "application/json"}
                if gemini_key.startswith("ya29."):
                    headers["Authorization"] = f"Bearer {gemini_key}"
                else:
                    url = f"{url}?key={gemini_key}"
                test_resp = requests.post(url, json={"contents": [{"parts": [{"text": "Hello"}]}]}, headers=headers, timeout=5)
                if test_resp.status_code == 200:
                    st.sidebar.success("🟢 Gemini connected successfully!")
                else:
                    st.sidebar.error(f"🔴 Connection failed: {test_resp.text}")
            except Exception as e:
                st.sidebar.error(f"🔴 Error: {e}")

    llm_fn = lambda r_text, jd, algo_res, _key=gemini_key, _model=gemini_model: judge_with_gemini(r_text, jd, algo_res, api_key=_key, model=_model)

elif llm_provider == "Simulated LLM (Offline)":
    st.sidebar.info("💡 Running offline qualitative simulation. No server or API keys required.")
    llm_fn = lambda r_text, jd, algo_res: judge_with_simulation(r_text, jd, algo_res)

# 3. Email SMTP Settings Panel
st.sidebar.markdown("---")
st.sidebar.subheader("📧 Email Outreach Settings")
outreach_mode = st.sidebar.selectbox(
    "Outreach Send Mode",
    ["Simulated Sandboxed Send", "SMTP Server Direct Send"],
    index=0,
    help="Select whether to simulate sending outreach or dispatch via SMTP server."
)

smtp_settings = {
    "mode": "Simulated" if outreach_mode == "Simulated Sandboxed Send" else "SMTP",
    "host": "smtp.gmail.com",
    "port": 587,
    "user": "",
    "password": "",
    "sender_name": "SmartATS Recruitment Team"
}

if outreach_mode == "SMTP Server Direct Send":
    smtp_settings["host"] = st.sidebar.text_input("SMTP Host", value="smtp.gmail.com")
    smtp_settings["port"] = st.sidebar.number_input("SMTP Port", min_value=1, max_value=65535, value=587)
    smtp_settings["user"] = st.sidebar.text_input("Sender Email Username", placeholder="e.g. recruit@company.com")
    smtp_settings["password"] = st.sidebar.text_input("SMTP Password", type="password", placeholder="e.g. app-specific password")
    smtp_settings["sender_name"] = st.sidebar.text_input("Sender Display Name", value="Acme Hiring Team")


# 4. Indian Hiring Mode Configurator
st.sidebar.markdown("---")
st.sidebar.subheader("🇮🇳 Indian Hiring Mode")
enable_indian_mode = st.sidebar.toggle(
    "Enable Indian Resume Processing",
    value=False,
    help="Apply Hinglish term normalization, Tier-1 college education boost, and Aadhaar/phone PII redaction."
)

# ---------- Main UI Structure ----------

# Header Banner
st.markdown(
    """
    <div class="header-banner">
        <h1>SmartATS — Explainable Hybrid Resume Ranker</h1>
        <p>A transparent, multi-dimensional candidate evaluation dashboard. Zero black-box score calculations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Helper function for routing ----------
def flatten_candidate(c):
    eval_data = c.get("evaluation") or {}
    return {**c, **eval_data}


def render_candidate_card_with_outreach(r, anonymized_mode, card_idx, is_eligible, email_subj_template, email_body_template):
    is_top = (r.get('rank', 1) == 1)
    card_class = "candidate-card top-rank" if is_top else "candidate-card"
    
    badge_html = '<span class="badge badge-success">🟢 Eligible</span>' if is_eligible else '<span class="badge badge-danger">🔴 Ineligible</span>'
    
    # Display contact info if not anonymized
    contact_html = ""
    if not anonymized_mode and r.get("contact"):
        email = r["contact"].get("email", "Not Provided")
        phone = r["contact"].get("phone", "Not Provided")
        contact_html = f"""<div style="margin-top: 0.5rem; font-size: 0.85rem; color: #CBD5E1; font-family: 'Inter';">📧 <strong>Email:</strong> <a href="mailto:{email}" style="color: #FBBF24; text-decoration: none; border-bottom: 1px dashed #FBBF24;">{email}</a> &nbsp;|&nbsp; 📞 <strong>Phone:</strong> <span style="color: #FBBF24;">{phone}</span></div>"""
    
    matched_role_html = f"""<div style="margin-top: 0.2rem; font-size: 0.85rem; color: #E2E8F0;">🎯 <strong>Matched Role:</strong> <span style="color: #FBBF24; font-weight: 600;">{r.get('best_fit_job_title', 'None')}</span></div>"""

    with st.container():
        st.markdown(
            f"""<div class="{card_class}">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div>
<h3 style="margin: 0; display: inline-block; vertical-align: middle;">#{r.get('rank', 1)} — {r['name']}</h3>
<div style="display: inline-block; margin-left: 0.5rem; vertical-align: middle;">{badge_html}</div>
<span class="badge badge-primary">{r.get('candidate_years', 0.0)} Years Relevant Exp</span>
{matched_role_html}
{contact_html}
</div>
<div style="text-align: right;">
<div style="font-size: 1.8rem; font-weight: 700; color: {'#34D399' if is_top else '#FBBF24'};">{r['final_score']}%</div>
<div style="font-size: 0.8rem; color: #CBD5E1;">Match Rating</div>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )
        
        # Action button row
        if not anonymized_mode:
            btn_label = "✉️ Send Interview Call" if is_eligible else "✉️ Send Sorry Mail"
            email_addr = r.get("contact", {}).get("email", "Not Provided")
            if email_addr not in ["Not Provided", "[REDACTED_EMAIL]"]:
                if st.button(btn_label, key=f"outreach_{r['filename']}_{card_idx}"):
                    # Send outreach email
                    subj = email_subj_template.replace("{name}", r["name"]).replace("{score}", f"{r['final_score']}%").replace("{role}", r.get("best_fit_job_title", ""))
                    body = email_body_template.replace("{name}", r["name"]).replace("{score}", f"{r['final_score']}%").replace("{role}", r.get("best_fit_job_title", ""))
                    with st.spinner("Dispatching email..."):
                        res = send_and_log_email(email_addr, subj, body, smtp_settings, r["name"])
                    
                    if res["success"]:
                        mode_text = "(Simulated)" if res.get("mode") == "Simulated" else "(SMTP)"
                        st.toast(f"🎉 Outreach successfully sent to {r['name']} {mode_text}!", icon="✅")
                    else:
                        st.error(f"❌ Failed to send email to {r['name']}: {res['error']}")
            else:
                st.warning("⚠️ No valid email address for outreach.")
        
        # Detailed breakdown section
        with st.expander(f"Inspect scoring breakdown and feedback details for {r['name']}"):
            c1, c2 = st.columns([1, 1.2])
            
            with c1:
                # Radar chart
                radar_fig = go.Figure()
                breakdown = r.get("breakdown") or {"skill_score": 0, "experience_score": 0, "education_score": 0, "semantic_score": 0}
                radar_fig.add_trace(go.Scatterpolar(
                    r=[
                        breakdown.get("skill_score", 0),
                        breakdown.get("experience_score", 0),
                        breakdown.get("education_score", 0),
                        breakdown.get("semantic_score", 0)
                    ],
                    theta=["Skills Match", "Experience Match", "Education Match", "Semantic Similarity"],
                    fill='toself',
                    name=r["name"],
                    line_color='#EF4444',
                    fillcolor='rgba(239, 68, 68, 0.15)'
                ))
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True, 
                            range=[0, 100],
                            gridcolor='rgba(220, 38, 38, 0.15)',
                            linecolor='rgba(220, 38, 38, 0.25)',
                            tickfont=dict(color='#94A3B8')
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(220, 38, 38, 0.15)',
                            tickfont=dict(color='#F8FAFC', size=9)
                        ),
                        bgcolor='rgba(30, 16, 16, 0.45)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    height=320,
                    margin=dict(t=20, b=20, l=40, r=40)
                )
                st.plotly_chart(radar_fig, use_container_width=True, key=f"radar_{r['name']}_{card_idx}")
                
            with c2:
                st.markdown("#### Detail Metrics Summary")
                
                # Mandatory skills penalty warning
                if r.get("mandatory_penalty", 0) > 0:
                    st.warning(
                        f"⚠️ **Mandatory Gate Penalty:** Candidate is missing required skills: "
                        f"**{', '.join(r.get('missing_mandatory_skills', []))}**. Applied penalty: -{r['mandatory_penalty']} pts."
                    )
                
                # Highlight lists
                st.markdown("**Matched Technical Skills:**")
                matched = r.get("matched_skills") or []
                if matched:
                    skill_html = "".join([f'<span class="badge badge-success">{s}</span>' for s in matched])
                    st.markdown(f'<div style="margin-bottom: 0.5rem;">{skill_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("_No direct skill matches found._")
                    
                st.markdown("**Missing Required Skills:**")
                missing = r.get("missing_skills") or []
                if missing:
                    skill_html = "".join([f'<span class="badge badge-danger">{s}</span>' for s in missing])
                    st.markdown(f'<div style="margin-bottom: 0.5rem;">{skill_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge badge-success">Full JD Skill Coverage!</span>', unsafe_allow_html=True)
                    
                st.write(f"💼 **Experience Profile:** {r.get('candidate_years', 0.0)} yrs relevant vs {r.get('required_years', 0.0)} yrs required.")
                
                deg_labels = {5: "PhD/Doctorate", 4: "Master's Degree", 3: "Bachelor's Degree", 2: "Diploma", 1: "High School", 0: "None"}
                cand_deg = deg_labels.get(r.get('candidate_degree_level', 0), "Undergraduate/Other")
                req_deg = deg_labels.get(r.get('required_degree_level', 0), "No degree requirement")
                st.write(f"🎓 **Education Level:** {cand_deg} (Requirement: {req_deg})")
                if r.get("education_boost_applied"):
                    st.info("🎓 **Tier-1 College Bonus:** +10 points applied for Indian Tier-1 Institution (IIT/NIT/IIIT/BITS/etc.).")
                
                # LLM Judge response if enabled
                if r.get("llm_review"):
                    llm_rev = r["llm_review"]
                    if "error" in llm_rev:
                        st.error(f"LLM Judge Layer Error: {llm_rev['error']}")
                    else:
                        nudge = llm_rev.get("score_adjustment", 0)
                        sign = "+" if nudge >= 0 else ""
                        adj_color = "#34D399" if nudge >= 0 else "#EF4444"
                        
                        # Set color and label for eligibility status
                        elig_val = llm_rev.get("eligibility", "potentially_eligible")
                        elig_label = elig_val.replace("_", " ").title()
                        if elig_val == "eligible":
                            elig_color = "#059669"
                        elif elig_val == "potentially_eligible":
                            elig_color = "#D97706"
                        else:
                            elig_color = "#DC2626"
                        
                        st.markdown("---")
                        st.markdown(
                            f"""<div class="llm-judge-box">
<div style="display: flex; justify-content: space-between; align-items: center;">
<strong style="color: #F8FAFC;">🧠 LLM Judge Assessment</strong>
<div>
<span style="background-color: {elig_color}15; color: {elig_color}; border: 1px solid {elig_color}30; padding: 0.2rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; margin-right: 0.5rem;">
{elig_label}
</span>
<span style="color: {adj_color}; font-weight: 700;">{sign}{nudge} pts adjustment</span>
</div>
</div>
<p style="margin-top: 0.5rem; margin-bottom: 0; color: #CBD5E1; font-style: italic;">
"{llm_rev.get('justification', '')}"
</p>
</div>""",
                            unsafe_allow_html=True
                        )
                        
                        # Render the Candidate summary
                        if llm_rev.get("summary"):
                            st.markdown(
                                f"""<div style="background-color: rgba(30, 16, 16, 0.35); border: 1px solid rgba(220, 38, 38, 0.2); border-radius: 12px; padding: 1.2rem; margin-top: 1rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);">
<strong style="color: #FBBF24; font-family: 'Inter'; font-size: 1rem; display: flex; align-items: center;">
📝 Candidate Profile Summary
</strong>
<p style="margin-top: 0.5rem; margin-bottom: 0; color: #CBD5E1; font-size: 0.9rem; line-height: 1.5; font-style: normal;">
{llm_rev['summary']}
</p>
</div>""",
                                unsafe_allow_html=True
                            )
                        
            # Raw Resume Snippet
            with st.expander("Preview raw resume text reviewed by engine"):
                st.text_area("Resume Content", value=r.get("text", ""), height=250, disabled=True, key=f"raw_res_{r['name']}_{card_idx}")


def run_routing_pipeline(anonymize, enable_indian_mode):
    openings = load_all_job_openings()
    if not openings:
        st.error("No active job openings found in company repository. Please manage job openings in Tab 2 first.")
        return
    
    if not st.session_state.get("resume_pool"):
        try:
            hackathon_candidates = load_hackathon_candidate_pool()
        except (FileNotFoundError, ValueError):
            hackathon_candidates = []
        if hackathon_candidates:
            st.session_state["resume_pool"] = hackathon_candidates
            st.session_state["needs_routing"] = True
        else:
            st.info("Candidate pool is empty. Please upload resumes or load demo resumes.")
            return
        
    with st.status("🤖 SmartATS AI Agent: Screening and Auto-Routing Candidates...", expanded=True) as status:
        resumes_to_score = []
        status.write("🔍 **Step 1:** Preparing candidate pool and checking PII settings...")
        
        for idx, item in enumerate(st.session_state["resume_pool"], start=1):
            raw_text = item["text"]
            cand_name = item["name"]
            contact = item["contact"]
            
            # Apply Indian PII Redaction if Indian Hiring Mode is active
            if enable_indian_mode:
                from engine.india_taxonomy import redact_indian_pii
                raw_text = redact_indian_pii(raw_text)
                if not anonymize:
                    email = contact.get("email", "Not Provided")
                    phone = contact.get("phone", "Not Provided")
                    phone = redact_indian_pii(phone)
                    contact = {"email": email, "phone": phone}
            
            if anonymize:
                scrubbed_text = redact_pii(raw_text)
                cand_name = f"Candidate #{idx} (Anonymous)"
                resumes_to_score.append({
                    "name": cand_name,
                    "filename": item["filename"],
                    "text": scrubbed_text,
                    "contact": {"email": "[REDACTED_EMAIL]", "phone": "[REDACTED_PHONE]"}
                })
            else:
                resumes_to_score.append({
                    "name": cand_name,
                    "filename": item["filename"],
                    "text": raw_text,
                    "contact": contact
                })
        
        status.write("🛠️ **Step 2:** Matching candidates against active roles in parallel...")
        status.write("💼 **Step 3:** Analyzing experience timelines & gating degree levels...")
        
        embedding_fn = None
        if llm_provider == "Google Gemini (Cloud)" and gemini_key:
            status.write("🧠 **Step 4 (Advanced):** Using Gemini `text-embedding-004` API for semantic similarity vector mapping...")
            embedding_fn = lambda r_text, jd_text, _key=gemini_key: calculate_gemini_embedding_similarity(r_text, jd_text, _key)
        else:
            status.write("🧠 **Step 4:** Computing TF-IDF vector document semantic similarity matrix...")
            
        status.write(f"🤖 **Step 5:** Invoking {llm_provider} qualitative judge layers...")
        
        routed_results = route_resumes_to_open_jobs(
            resumes=resumes_to_score,
            open_jobs=openings,
            llm_judge_fn=llm_fn,
            weights=weights_dict,
            enable_indian_mode=enable_indian_mode,
            embedding_fn=embedding_fn
        )
        
        # Add contact info back to routed results for display
        for cat in ["eligible", "ineligible"]:
            for r in routed_results[cat]:
                orig_item = next((item for item in st.session_state["resume_pool"] if item["filename"] == r["filename"]), None)
                if orig_item:
                    if anonymize:
                        r["contact"] = {"email": "[REDACTED_EMAIL]", "phone": "[REDACTED_PHONE]"}
                    else:
                        r["contact"] = orig_item["contact"]
                        if enable_indian_mode:
                            from engine.india_taxonomy import redact_indian_pii
                            r["contact"]["phone"] = redact_indian_pii(r["contact"].get("phone", "Not Provided"))
                else:
                    r["contact"] = {"email": "Not Provided", "phone": "Not Provided"}
                    
        st.session_state["routed_results"] = routed_results
        st.session_state["routed_anonymized_mode"] = anonymize
        st.session_state["routed_indian_mode"] = enable_indian_mode
        status.update(label="✅ Auto-Routing Complete!", state="complete", expanded=False)


# ---------- Main UI Structure ----------
tab_router, tab_jobs, tab_audit, tab_outreach = st.tabs(["🚀 Automated Candidate Router", "🏢 Manage Job Openings", "📊 Diversity & Bias Audit", "📧 Outreach Dispatch Log"])

# TAB 1: AUTOMATED CANDIDATE ROUTER
with tab_router:
    st.markdown("## 🚀 Candidate Matching & Routing Dashboard")
    st.markdown("Upload resumes to automatically screen and match them against all currently active job openings.")
    
    col_up, col_opts = st.columns([1.5, 1])
    
    with col_up:
        # Initialize session state for candidate pool
        if "resume_pool" not in st.session_state:
            st.session_state["resume_pool"] = []
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = set()
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0
            
        uploaded_resumes = st.file_uploader(
            "Upload Resumes (PDF, DOCX, TXT) — Multiple Supported",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}"
        )
        
        # Process newly uploaded files into the pool
        if uploaded_resumes:
            new_added = False
            for f in uploaded_resumes:
                file_id = f"{f.name}_{f.size}"
                if file_id not in st.session_state["processed_files"]:
                    try:
                        raw_text = extract_text(f.name, f.read())
                        cand_name = extract_candidate_name(raw_text, fallback=f.name)
                        contact = extract_contact_info(raw_text)
                        
                        # Prevent duplicates in the pool
                        if not any(item["filename"] == f.name for item in st.session_state["resume_pool"]):
                            st.session_state["resume_pool"].append({
                                "name": cand_name,
                                "filename": f.name,
                                "text": raw_text,
                                "contact": contact
                            })
                        st.session_state["processed_files"].add(file_id)
                        new_added = True
                    except Exception as e:
                        st.error(f"Error parsing {f.name}: {e}")
            if new_added:
                st.session_state["needs_routing"] = True
                st.toast("Added new resumes to candidate pool!", icon="✅")
                st.rerun()

    with col_opts:
        use_demo_resumes = st.checkbox(
            "⚡ Load Sample Demo Resumes", 
            value=False, 
            help="Load preloaded resumes (Strong, Moderate, and Weak match) instantly for testing."
        )
        if use_demo_resumes:
            st.info("💡 **Sample Demo Resumes:** Pre-packaged candidate resumes to test features. Uncheck to evaluate only your own uploads.")
        
        demo_filenames = {
            "aditi_sharma_strong_match.txt", "priya_nair_weak_match.txt", "rahul_verma_moderate_match.txt",
            "karan_mehta_frontend_strong.txt", "arjun_reddy_devops_strong.txt", "dr_neha_patel_data_science_strong.txt",
            "vikram_singh_pm_strong.txt", "rachel_green_security_strong.txt", "amit_patel_mobile_strong.txt"
        }
        
        if use_demo_resumes:
            demo_files = get_sample_resumes()
            demo_added = False
            for item in demo_files:
                file_id = f"demo_{item['filename']}"
                if file_id not in st.session_state["processed_files"]:
                    raw_text = item["text"]
                    cand_name = extract_candidate_name(raw_text, fallback=item["filename"])
                    contact = extract_contact_info(raw_text)
                    st.session_state["resume_pool"].append({
                        "name": cand_name,
                        "filename": item["filename"],
                        "text": raw_text,
                        "contact": contact
                    })
                    st.session_state["processed_files"].add(file_id)
                    demo_added = True
            if demo_added:
                st.session_state["needs_routing"] = True
                st.rerun()
        else:
            # Remove demo resumes if unchecked
            orig_len = len(st.session_state["resume_pool"])
            st.session_state["resume_pool"] = [
                r for r in st.session_state["resume_pool"] if r["filename"] not in demo_filenames
            ]
            st.session_state["processed_files"] = {
                fid for fid in st.session_state["processed_files"] if not any(df in fid for df in demo_filenames)
            }
            if len(st.session_state["resume_pool"]) != orig_len:
                st.session_state["needs_routing"] = True
                st.rerun()
                
        anonymize = st.checkbox(
            "Anonymize Candidate Information (PII Scrubbing)",
            value=False,
            help="Scrub candidates' names, emails, and phone numbers before scoring."
        )

    # Manage Active Pool Expander
    if st.session_state["resume_pool"]:
        with st.expander(f"📁 Manage Candidate Pool ({len(st.session_state['resume_pool'])} Resumes Active)", expanded=False):
            st.markdown("<div style='max-height: 250px; overflow-y: auto;'>", unsafe_allow_html=True)
            for i, r in enumerate(list(st.session_state["resume_pool"])):
                c_name, c_act = st.columns([4, 1])
                with c_name:
                    st.markdown(f"📄 **{r['name']}** <span style='font-size: 0.75rem; color: #94A3B8;'>({r['filename']})</span>", unsafe_allow_html=True)
                with c_act:
                    if st.button("🗑️", key=f"del_{r['filename']}_{i}", help=f"Remove {r['name']} from pool"):
                        st.session_state["resume_pool"].pop(i)
                        file_id_to_remove = None
                        for fid in st.session_state["processed_files"]:
                            if r["filename"] in fid:
                                file_id_to_remove = fid
                                break
                        if file_id_to_remove:
                            st.session_state["processed_files"].remove(file_id_to_remove)
                        st.session_state["uploader_key"] += 1
                        st.session_state["needs_routing"] = True
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("🗑️ Clear Entire Pool", type="secondary", use_container_width=True):
                st.session_state["resume_pool"] = []
                st.session_state["processed_files"] = set()
                st.session_state["uploader_key"] += 1
                st.session_state["routed_results"] = None
                st.session_state["needs_routing"] = True
                st.rerun()

    st.markdown("---")
    
    col_trigger, col_info = st.columns([1, 2])
    with col_trigger:
        run_btn = st.button("🚀 Run Auto-Router Scorer", type="primary", use_container_width=True)
        if run_btn:
            st.session_state["needs_routing"] = True
            st.rerun()
            
    with col_info:
        st.info("💡 Candidate matching runs automatically on resume uploads. Click the button to force re-run if weights or LLM settings change.")
        
    # Check if we should automatically run
    if st.session_state.get("needs_routing", False):
        st.session_state["needs_routing"] = False
        run_routing_pipeline(anonymize, enable_indian_mode)
        st.rerun()
        
    if st.session_state.get("routed_results") is None and st.session_state.get("resume_pool") and load_all_job_openings():
        run_routing_pipeline(anonymize, enable_indian_mode)
        st.rerun()
        
    # If routed results exist, show them
    if st.session_state.get("routed_results"):
        routed = st.session_state["routed_results"]
        anonymized_mode = st.session_state.get("routed_anonymized_mode", False)
        
        eligible = [flatten_candidate(c) for c in routed["eligible"]]
        ineligible = [flatten_candidate(c) for c in routed["ineligible"]]
        eligible.sort(key=lambda x: (-float(x.get("final_score", 0.0)), x.get("filename", ""), x.get("name", "")))
        ineligible.sort(key=lambda x: (-float(x.get("final_score", 0.0)), x.get("filename", ""), x.get("name", "")))

        for idx, candidate in enumerate(eligible, start=1):
            candidate["rank"] = idx
        for idx, candidate in enumerate(ineligible, start=1):
            candidate["rank"] = idx
        
        st.markdown("### 🔍 Search & Interactive Filters")
        sf_col1, sf_col2, sf_col3 = st.columns(3)
        with sf_col1:
            search_query = st.text_input("👤 Search by Name or Skill", "", placeholder="e.g. Aditi, Python, AWS...", key="router_search_query")
        with sf_col2:
            min_score = st.slider("🎯 Min Match Score %", min_value=0, max_value=100, value=0, step=5, key="router_min_score")
        with sf_col3:
            min_exp = st.number_input("💼 Min Years Exp", min_value=0.0, max_value=20.0, value=0.0, step=0.5, key="router_min_exp")
            
        # Apply filters
        filtered_eligible = []
        for r in eligible:
            query_match = True
            if search_query:
                q = search_query.lower()
                name_match = q in r["name"].lower()
                skill_match = any(q in s.lower() for s in r.get("matched_skills", []))
                filename_match = q in r["filename"].lower()
                query_match = name_match or skill_match or filename_match
            score_match = r["final_score"] >= min_score
            exp_match = r.get("candidate_years", 0.0) >= min_exp
            if query_match and score_match and exp_match:
                filtered_eligible.append(r)
                
        filtered_ineligible = []
        for r in ineligible:
            query_match = True
            if search_query:
                q = search_query.lower()
                name_match = q in r["name"].lower()
                skill_match = any(q in s.lower() for s in r.get("matched_skills", []))
                filename_match = q in r["filename"].lower()
                query_match = name_match or skill_match or filename_match
            score_match = r["final_score"] >= min_score
            exp_match = r.get("candidate_years", 0.0) >= min_exp
            if query_match and score_match and exp_match:
                filtered_ineligible.append(r)
                
        # Display top-level metric summaries
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Candidates</div>
                    <div class="metric-value">{len(eligible) + len(ineligible)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Eligible Candidates</div>
                    <div class="metric-value">{len(filtered_eligible)} / {len(eligible)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Ineligible Candidates</div>
                    <div class="metric-value">{len(filtered_ineligible)} / {len(ineligible)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Scoring Mode</div>
                    <div class="metric-value" style="font-size: 1.1rem; padding-top: 0.6rem; color: #FBBF24;">
                        {'🔒 Anonymized' if anonymized_mode else '👤 Standard'}<br>
                        <span style="font-size: 0.75rem; color: #94A3B8;">LLM: {llm_provider}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Export all candidate results to CSV
        all_candidates = []
        for r in eligible:
            r_copy = r.copy()
            r_copy["status"] = "Eligible"
            all_candidates.append(r_copy)
        for r in ineligible:
            r_copy = r.copy()
            r_copy["status"] = "Ineligible"
            all_candidates.append(r_copy)
            
        all_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        all_candidates_csv = []
        for idx, r in enumerate(all_candidates, 1):
            breakdown = r.get("breakdown") or {}
            all_candidates_csv.append({
                "Rank": idx,
                "Name": r.get("name", "Unknown"),
                "Final Score": f"{r.get('final_score', 0.0)}%",
                "Matched Role": r.get("best_fit_job_title", "None"),
                "Skill Match": f"{breakdown.get('skill_score', 0.0)}%",
                "Experience Match": f"{breakdown.get('experience_score', 0.0)}%",
                "Education Match": f"{breakdown.get('education_score', 0.0)}%",
                "Semantic Similarity": f"{breakdown.get('semantic_score', 0.0)}%",
                "Years Experience": r.get("candidate_years", 0.0),
                "Status": r.get("status", ""),
                "Email": r.get("contact", {}).get("email", "Not Provided")
            })
            
        if all_candidates_csv:
            df_all = pd.DataFrame(all_candidates_csv)
            csv_data_all = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export All Candidate Results to CSV",
                data=csv_data_all,
                file_name="smartats_ranking_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_all_ranking_results"
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sub-tabs for eligible vs ineligible
        sub_tab_el, sub_tab_in = st.tabs([
            f"🟢 Routed Eligible Candidates ({len(filtered_eligible)})",
            f"🔴 Ineligible Candidates ({len(filtered_ineligible)})"
        ])
        
        # Template bodies
        default_eligible_body = (
            "Hi {name},\n\n"
            "We reviewed your resume and were highly impressed by your background. "
            "Your match score is {score} and you are matched for our {role} opening.\n\n"
            "We would love to schedule a brief call to discuss this opportunity further. "
            "Please let us know your availability this week.\n\n"
            "Best regards,\n"
            + smtp_settings['sender_name']
        )
        
        default_ineligible_body = (
            "Hi {name},\n\n"
            "Thank you for your interest in the {role} position. "
            "While we were impressed by your background, we have decided to move forward with other candidates whose profiles more closely align with our current technical requirements.\n\n"
            "We will keep your profile in our candidate pool for future roles. We wish you the best in your job search.\n\n"
            "Best regards,\n"
            + smtp_settings['sender_name']
        )
        
        with sub_tab_el:
            if not filtered_eligible:
                st.info("No eligible candidates found matching the criteria.")
            else:
                if anonymized_mode:
                    st.warning("⚠️ PII Redaction is active. Candidate names and emails are masked, so email outreach is disabled. Uncheck PII Redaction to enable outreach.")
                else:
                    st.markdown("### ✉️ Send Email Outreach Campaign")
                    
                    # Collapsible Template
                    el_subj_template = st.text_input("Interview Subject Line Template", value="Interview Invitation: {role} Role", key="el_subject_tmpl")
                    el_body_template = st.text_area("Interview Email Template Body", value=default_eligible_body, height=180, key="el_body_tmpl")
                    st.caption("Placeholders: `{name}`, `{score}`, `{role}`")
                    
                    # Recipient email list
                    valid_recipients = [r for r in filtered_eligible if r.get("contact") and r["contact"].get("email") not in ["Not Provided", "[REDACTED_EMAIL]"]]
                    
                    # Bulk send
                    if valid_recipients:
                        st.caption(f"Ready to send to {len(valid_recipients)} candidates.")
                        if st.button("🚀 Bulk Send Interview Calls", key="send_bulk_el_emails", use_container_width=True):
                            import concurrent.futures
                            
                            status_container = st.container()
                            status_container.info("🔄 Sending outreach emails in parallel...")
                            
                            def dispatch_single(cand):
                                subj = el_subj_template.replace("{name}", cand["name"]).replace("{score}", f"{cand['final_score']}%").replace("{role}", cand.get("best_fit_job_title", ""))
                                body = el_body_template.replace("{name}", cand["name"]).replace("{score}", f"{cand['final_score']}%").replace("{role}", cand.get("best_fit_job_title", ""))
                                res = send_and_log_email(cand["contact"]["email"], subj, body, smtp_settings, cand["name"])
                                return cand["name"], cand["contact"]["email"], res
                                
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                results = list(executor.map(dispatch_single, valid_recipients))
                                
                            status_container.empty()
                            st.success(f"🎉 Outreach Campaign complete! Sent to {len(valid_recipients)} candidates.")
                            
                            report_rows = []
                            for name, email, res in results:
                                icon = "white_check_mark" if res["success"] else "x"
                                detail = "Sent (Simulated)" if res.get("mode") == "Simulated" else ("Sent (SMTP)" if res["success"] else res["error"])
                                report_rows.append(f" {icon} **{name}** ({email}): {detail}")
                            st.markdown("\n".join(report_rows))
                
                # Group eligible candidates by matched job title
                grouped_eligible = {}
                for r in filtered_eligible:
                    title = r.get("best_fit_job_title", "Unmatched")
                    if title not in grouped_eligible:
                        grouped_eligible[title] = []
                    grouped_eligible[title].append(r)
                
                # Render role groups in score order while keeping the existing layout intact.
                ordered_role_titles = sorted(
                    grouped_eligible.keys(),
                    key=lambda title: (-max((float(c.get("final_score", 0.0)) for c in grouped_eligible[title]), default=0.0), title),
                )
                for role_title in ordered_role_titles:
                    role_cands = sorted(
                        grouped_eligible[role_title],
                        key=lambda x: (-float(x.get("final_score", 0.0)), x.get("filename", ""), x.get("name", "")),
                    )
                    st.markdown(f"#### 💼 {role_title} ({len(role_cands)} Candidates)")

                    for idx, c in enumerate(role_cands, start=1):
                        subj_tmpl = el_subj_template if not anonymized_mode else ""
                        body_tmpl = el_body_template if not anonymized_mode else ""
                        render_candidate_card_with_outreach(c, anonymized_mode, f"el_{idx}_{role_title.replace(' ', '_')}", True, subj_tmpl, body_tmpl)
                        
        with sub_tab_in:
            if not filtered_ineligible:
                st.info("No ineligible candidates found matching the criteria.")
            else:
                if anonymized_mode:
                    st.warning("⚠️ PII Redaction is active. Candidate names and emails are masked, so email outreach is disabled. Uncheck PII Redaction to enable outreach.")
                else:
                    st.markdown("### ✉️ Send Candidate Status Updates")
                    
                    in_subj_template = st.text_input("Rejection Subject Line Template", value="Application Update: {role} Role", key="in_subject_tmpl")
                    in_body_template = st.text_area("Rejection Email Template Body", value=default_ineligible_body, height=180, key="in_body_tmpl")
                    st.caption("Placeholders: `{name}`, `{score}`, `{role}`")
                    
                    valid_recipients = [r for r in filtered_ineligible if r.get("contact") and r["contact"].get("email") not in ["Not Provided", "[REDACTED_EMAIL]"]]
                    
                    if valid_recipients:
                        st.caption(f"Ready to send to {len(valid_recipients)} candidates.")
                        if st.button("🚀 Bulk Send Rejection Emails", key="send_bulk_in_emails", use_container_width=True):
                            import concurrent.futures
                            
                            status_container = st.container()
                            status_container.info("🔄 Sending status updates in parallel...")
                            
                            def dispatch_single(cand):
                                subj = in_subj_template.replace("{name}", cand["name"]).replace("{score}", f"{cand['final_score']}%").replace("{role}", cand.get("best_fit_job_title", ""))
                                body = in_body_template.replace("{name}", cand["name"]).replace("{score}", f"{cand['final_score']}%").replace("{role}", cand.get("best_fit_job_title", ""))
                                res = send_and_log_email(cand["contact"]["email"], subj, body, smtp_settings, cand["name"])
                                return cand["name"], cand["contact"]["email"], res
                                
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                results = list(executor.map(dispatch_single, valid_recipients))
                                
                            status_container.empty()
                            st.success(f"🎉 Notification complete! Sent to {len(valid_recipients)} candidates.")
                            
                            report_rows = []
                            for name, email, res in results:
                                icon = "white_check_mark" if res["success"] else "x"
                                detail = "Sent (Simulated)" if res.get("mode") == "Simulated" else ("Sent (SMTP)" if res["success"] else res["error"])
                                report_rows.append(f" {icon} **{name}** ({email}): {detail}")
                            st.markdown("\n".join(report_rows))
                
                # List ineligible candidates
                for idx, c in enumerate(sorted(filtered_ineligible, key=lambda x: (-float(x.get("final_score", 0.0)), x.get("filename", ""), x.get("name", ""))), start=1):
                    c["rank"] = idx
                    subj_tmpl = in_subj_template if not anonymized_mode else ""
                    body_tmpl = in_body_template if not anonymized_mode else ""
                    render_candidate_card_with_outreach(c, anonymized_mode, f"in_{idx}", False, subj_tmpl, body_tmpl)

        # Export CSV
        csv_rows = []
        for r in filtered_eligible + filtered_ineligible:
            csv_rows.append({
                "Candidate": r["name"],
                "Email": r.get("contact", {}).get("email", "Not Provided"),
                "Phone": r.get("contact", {}).get("phone", "Not Provided"),
                "Final Score": f"{r['final_score']}%",
                "Best Fit Job": r.get("best_fit_job_title", "None"),
                "Eligibility": "Eligible" if r.get("is_eligible") else "Ineligible",
                "Skills Match": f"{r.get('breakdown', {}).get('skill_score', 0)}%",
                "Experience Match": f"{r.get('breakdown', {}).get('experience_score', 0)}%",
                "Education Match": f"{r.get('breakdown', {}).get('education_score', 0)}%",
                "Semantic Sim": f"{r.get('breakdown', {}).get('semantic_score', 0)}%",
                "Years Exp": r.get("candidate_years", 0.0)
            })
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Shortlist to CSV", csv_data, "smartats_routed_shortlist.csv", "text/csv", use_container_width=True)
    else:
        st.info("Upload resumes to trigger candidate routing across active job openings.")

# TAB 2: MANAGE JOB OPENINGS
with tab_jobs:
    st.markdown("## 🏢 Manage Job Openings")
    st.markdown("Create and save job openings permanently. These roles are used to auto-match and route uploaded resumes.")
    
    col_add, col_list = st.columns([1, 1.2])
    
    with col_add:
        st.markdown("### ➕ Create New Job Opening")
        with st.form("add_job_form", clear_on_submit=True):
            job_title = st.text_input("Job Title", placeholder="e.g. Senior Frontend React Developer")
            req_years = st.number_input("Required Experience (Years)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
            
            deg_map_options = ["None", "High School", "Diploma", "Bachelor's Degree", "Master's Degree", "PhD/Doctorate"]
            req_deg_label = st.selectbox("Required Education Level", deg_map_options, index=3) # Default to Bachelor's
            req_deg = deg_map_options.index(req_deg_label)
            
            mand_skills_str = st.text_input("Mandatory Skills (Comma separated)", placeholder="e.g. python, django, postgresql")
            
            jd_details = st.text_area("Full Job Description Details", height=200, placeholder="Paste details here describing roles and expectations...")
            
            submit_job = st.form_submit_button("💾 Save Position permanently")
            if submit_job:
                if not job_title.strip():
                    st.error("Job title is required.")
                elif not jd_details.strip():
                    st.error("Job description text details are required.")
                else:
                    # Parse skills
                    skills = [s.strip().lower() for s in mand_skills_str.split(",") if s.strip()]
                    slug = save_job_opening(
                        title=job_title.strip(),
                        jd_text=jd_details.strip(),
                        mandatory_skills=skills,
                        required_years=req_years,
                        required_degree=req_deg
                    )
                    st.session_state["needs_routing"] = True
                    st.toast(f"✅ Job Opening '{job_title}' successfully saved permanently!", icon="🎉")
                    st.rerun()

    with col_list:
        st.markdown("### 💼 Active Company Openings")
        openings = load_all_job_openings()
        if not openings:
            st.info("No active job openings found in repository.")
        else:
            deg_map_options = ["None", "High School", "Diploma", "Bachelor's Degree", "Master's Degree", "PhD/Doctorate"]
            for job in openings:
                with st.container():
                    # Card for each opening
                    st.markdown(
                        f"""<div class="candidate-card" style="margin-bottom: 1rem; border: 1px solid rgba(245, 158, 11, 0.25);">
<h4 style="margin: 0; color: #FBBF24;">{job['title']}</h4>
<div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.3rem;">
💼 <strong>Exp Required:</strong> {job['required_years']} Years &nbsp;|&nbsp; 
🎓 <strong>Edu Level:</strong> {deg_map_options[job['required_degree']]}
</div>
<div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.2rem;">
🎯 <strong>Mandatory Skills:</strong> {', '.join(job.get('mandatory_skills', [])) if job.get('mandatory_skills') else 'None Specified'}
</div>
</div>""",
                        unsafe_allow_html=True
                    )
                    # Delete action
                    if st.button("🗑️ Delete Position", key=f"del_job_{job['slug']}", help=f"Delete position: {job['title']}"):
                        if delete_job_opening(job['slug']):
                            st.session_state["needs_routing"] = True
                            st.toast(f"🗑️ Deleted position '{job['title']}'", icon="🗑️")
                            st.rerun()
                        else:
                            st.error("Failed to delete position.")

# TAB 3: DIVERSITY & BIAS AUDIT
with tab_audit:
    st.markdown("## Diversity, Equity & Inclusion (DE&I) Compliance Audit")
    st.caption("Verify that candidates are scored purely on skills and relevance, without age or credential bias.")
    
    if st.session_state.get("routed_results") is None:
        st.info("No audit data available. Please upload resumes and match them in Tab 1 first.")
    else:
        routed = st.session_state["routed_results"]
        eligible = [flatten_candidate(c) for c in routed["eligible"]]
        ineligible = [flatten_candidate(c) for c in routed["ineligible"]]
        all_routed = eligible + ineligible
        
        if not all_routed:
            st.info("No candidates in the pool to analyze.")
        else:
            # Prepare data fields
            audit_data = []
            for r in all_routed:
                grad_year = extract_graduation_year(r.get("text", ""))
                audit_data.append({
                    "name": r["name"],
                    "score": r["final_score"],
                    "degree_level": r.get("candidate_degree_level", 0),
                    "grad_year": grad_year,
                    "years_exp": r.get("candidate_years", 0.0)
                })
            audit_df = pd.DataFrame(audit_data)
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("### Graduation Year Correlation (Age Bias Check)")
                # Filter out records where graduation year was not found
                valid_grad = audit_df.dropna(subset=["grad_year"])
                
                if not valid_grad.empty and len(valid_grad) >= 2:
                    fig_grad = px.scatter(
                        valid_grad, x="grad_year", y="score",
                        text="name",
                        labels={"grad_year": "Graduation Year (Proxy for Age)", "score": "Final Match Score (%)"},
                        title="Score vs. Graduation Year"
                    )
                    fig_grad.update_traces(textposition='top center', marker=dict(size=12, color='#EF4444'))
                    fig_grad.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#F8FAFC',
                        xaxis=dict(gridcolor='rgba(220, 38, 38, 0.15)', linecolor='rgba(220, 38, 38, 0.25)'),
                        yaxis=dict(gridcolor='rgba(220, 38, 38, 0.15)', linecolor='rgba(220, 38, 38, 0.25)')
                    )
                    st.plotly_chart(fig_grad, use_container_width=True)
                    
                    # Pearson correlation
                    corr = valid_grad["grad_year"].corr(valid_grad["score"])
                    if abs(corr) < 0.3:
                        st.success(f"🟢 **Minimal Bias:** Low correlation between graduation year and score (r = {corr:.2f}). Evaluator is fair.")
                    else:
                        st.warning(f"⚠️ **Noticeable Correlation:** Correlation found between graduation year and score (r = {corr:.2f}). Verify that years of experience requirements are appropriate.")
                else:
                    st.info("Upload more resumes with explicit graduation dates to generate the Age Bias audit scatter plot.")
                    
            with col_c2:
                st.markdown("### Degree Level Breakdown (Credential Bias)")
                deg_map = {5: "PhD", 4: "Master's", 3: "Bachelor's", 2: "Diploma", 1: "High School", 0: "None"}
                audit_df["degree_name"] = audit_df["degree_level"].map(deg_map)
                
                fig_edu = px.box(
                    audit_df, x="degree_name", y="score",
                    points="all",
                    labels={"degree_name": "Highest Education Degree", "score": "Final Match Score (%)"},
                    title="Score Distribution by Credential Level"
                )
                fig_edu.update_traces(marker=dict(color='#F59E0B'))
                fig_edu.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#F8FAFC',
                    xaxis=dict(gridcolor='rgba(220, 38, 38, 0.15)', linecolor='rgba(220, 38, 38, 0.25)'),
                    yaxis=dict(gridcolor='rgba(220, 38, 38, 0.15)', linecolor='rgba(220, 38, 38, 0.25)')
                )
                st.plotly_chart(fig_edu, use_container_width=True)
                
            st.markdown("---")
            st.markdown("### Qualitative Action-Verb Analysis")
            st.caption("Analyzes the types of vocabulary candidates use in their achievements. Heavy reliance on ownership verbs usually signals strong qualifiers, while passive terms suggest supporting responsibilities.")
            
            # Calculate verb usage counts
            leadership_verbs = ["led", "managed", "founded", "established", "architected", "headed", "spearheaded", "designed", "optimized", "built"]
            passive_verbs = ["assisted", "helped", "participated", "supported", "contributed"]
            
            verb_counts = []
            for r in all_routed:
                text_l = r.get("text", "").lower()
                lead_cnt = sum(len(re.findall(r"\b" + v + r"\b", text_l)) for v in leadership_verbs)
                pass_cnt = sum(len(re.findall(r"\b" + v + r"\b", text_l)) for v in passive_verbs)
                verb_counts.append({
                    "Candidate": r["name"],
                    "Ownership Verbs": lead_cnt,
                    "Supportive Verbs": pass_cnt
                })
                
            verb_df = pd.DataFrame(verb_counts)
            fig_verbs = px.bar(
                verb_df, x="Candidate", y=["Ownership Verbs", "Supportive Verbs"],
                barmode="group",
                color_discrete_map={"Ownership Verbs": "#F59E0B", "Supportive Verbs": "#EF4444"},
                title="Action Vocabulary Frequency Across Profiles"
            )
            fig_verbs.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC',
                xaxis=dict(gridcolor='rgba(220, 38, 38, 0.15)', linecolor='rgba(220, 38, 38, 0.25)'),
                yaxis=dict(gridcolor='rgba(220, 38, 38, 0.15)', linecolor='rgba(220, 38, 38, 0.25)')
            )
            st.plotly_chart(fig_verbs, use_container_width=True)

# TAB 4: OUTREACH DISPATCH LOG
with tab_outreach:
    st.markdown("## 📧 Outreach Dispatch Log")
    st.markdown("Monitor and track status updates and candidate emails dispatched during this session.")
    
    if "outreach_log" not in st.session_state or not st.session_state["outreach_log"]:
        st.info("No outreach emails sent during this session yet. Run campaigns or send emails in Tab 1 first.")
    else:
        log_df = pd.DataFrame(st.session_state["outreach_log"])
        
        # Display log table nicely
        st.dataframe(log_df, use_container_width=True)
        
        # Export Log CSV button
        csv_log_data = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Outreach Log to CSV",
            data=csv_log_data,
            file_name="smartats_outreach_log.csv",
            mime="text/csv",
            key="download_outreach_log_btn",
            use_container_width=True
        )

st.markdown("---")
st.caption("SmartATS — Built with high-end modular heuristics, PII Anonymization, and custom-weights explainability. Local-first architecture.")
