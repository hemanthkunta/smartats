# 🎯 SmartATS — Explainable Hybrid Resume Ranker

> Built for [Hackathon Name] — an Applicant Tracking System that ranks candidates against a job description using a **transparent, hybrid scoring algorithm** instead of an opaque "LLM says 85%" black box.

## 🚀 The Problem

Recruiters manually screen hundreds of resumes per role. Most "AI resume ranker" tools just shove resume text + job description into an LLM prompt and ask for a score — fast to build, but **non-deterministic, unexplainable, and prone to hallucination**. A score with no reasoning isn't trustworthy enough to act on.

## 💡 Our Approach: Hybrid, Explainable Scoring

Instead of one black-box score, SmartATS computes **four independent, deterministic signals**, each fully explainable, then blends them:

| Signal | Weight | What it measures |
|---|---|---|
| 🛠️ **Skill Match** | 40% | Taxonomy-aware overlap between JD-required skills and resume skills (handles synonyms: "JS" = "JavaScript", "ML" = "Machine Learning") |
| 💼 **Experience Match** | 25% | Years of experience extracted from resume, **gated by skill relevance** (irrelevant years count less) |
| 🎓 **Education Match** | 15% | Degree level required vs. degree level held |
| 🧠 **Semantic Similarity** | 20% | TF-IDF cosine similarity between full resume and JD text — catches contextual fit beyond keyword lists |

### The key differentiator: Relevance-Gated Experience

A candidate with 10 years of experience in an unrelated field shouldn't score 100% on "experience" for a role requiring 3 years of Python. Our engine scales the raw years-match score by how much of the candidate's skill set actually overlaps with the JD — so experience is only rewarded when it's *relevant* experience.

### Optional LLM Judge Layer (Local, via Ollama)

For nuance a deterministic algorithm can't capture (e.g. "led a 5-person team shipping a product" vs. "was a team member"), SmartATS can optionally call a **local LLM via Ollama** to review the top candidates and suggest a score adjustment.

This adjustment is **bounded to ±10 points** and always shown with its justification — the LLM can refine the score, but it can never override the explainable algorithmic core. This is a deliberate reliability choice: it gets the benefit of LLM judgment without the risk of hallucination wrecking the ranking.

**100% local. No resume data ever leaves your machine** — no OpenAI/external API calls required to run the core engine.

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Streamlit UI    │────▶│  Parser Layer     │────▶│  Scoring Engine      │
│  (file uploads,  │     │  (PDF/DOCX/TXT    │     │  - Skill taxonomy    │
│  results table,  │     │  extraction +     │     │  - TF-IDF cosine sim │
│  radar charts)   │     │  text cleaning)   │     │  - Experience/Edu    │
└─────────────────┘     └──────────────────┘     │    extraction        │
                                                    └──────────┬──────────┘
                                                               │
                                                               ▼
                                                  ┌─────────────────────┐
                                                  │  Optional LLM Judge  │
                                                  │  (Ollama, local,     │
                                                  │  bounded ±10 nudge)  │
                                                  └─────────────────────┘
```

## 📦 Tech Stack

- **Frontend:** Streamlit
- **Parsing:** PyPDF2, python-docx
- **Scoring:** Pure Python (no heavy ML deps — fast install, no GPU needed) using a custom skill taxonomy + lightweight TF-IDF cosine similarity
- **Optional LLM layer:** Ollama (local), default model `llama3.2:1b` (runs comfortably on 8GB RAM)
- **Visualization:** Plotly (radar charts, bar charts)

## 🛠️ Setup & Run

```bash
git clone https://github.com/hemanthkunta/smartats.git
cd smartats
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`.

### Optional: Enable the LLM Judge Layer

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2:1b
ollama serve
```

Then toggle "Enable LLM Judge Layer" in the app sidebar.

## 📂 Project Structure

```
smartats/
├── app.py                  # Streamlit UI
├── parsers/
│   └── parser.py            # PDF/DOCX/TXT text extraction + cleaning
├── engine/
│   ├── taxonomy.py          # Skill synonym map, section headers, degree levels
│   ├── scorer.py             # Core hybrid scoring algorithm
│   └── llm_judge.py          # Optional Ollama-based qualitative judge
├── data/
│   ├── sample_jd/            # Sample job description for demo
│   └── sample_resumes/       # 3 sample resumes (strong/moderate/weak match)
└── requirements.txt
```

## 🎬 Demo Flow

1. Paste or upload a Job Description
2. Upload multiple candidate resumes (PDF/DOCX/TXT)
3. Click "Rank Candidates"
4. View ranked results table + bar chart comparison
5. Expand any candidate for a full breakdown: radar chart of the 4 scoring dimensions, matched/missing skills, and (if enabled) the LLM judge's note

Sample data included in `data/` for an instant demo — no need to source real resumes.

## 🔮 Future Work

- Resume section-aware parsing (skills section vs. experience section weighting)
- Multi-JD batch comparison (rank one candidate pool against several open roles)
- Bias auditing pass (flag scoring patterns correlated with non-job-relevant resume features)
- Fine-tuned local embedding model swap-in for semantic similarity (currently TF-IDF for speed/zero-dependency reasons)

## 👤 Author

Built by Hemanth for Redrob x H2S — INDIA.RUNS Hackathon, June–July 2026. Track: Track 1 — Data & AI Challenge.

## 🇮🇳 Built for India

SmartATS has been tailor-made for the unique characteristics of Indian hiring:
- **Indian Resume Formats & Degree Variants:** Recognizes variants of `B.Tech` / `M.Tech` degrees and maps them to appropriate educational milestones.
- **IIT/NIT Academic Mapping:** Recognizes prestigious Indian educational institution names (IITs, NITs, BITS, etc.) to evaluate qualification levels.
- **Hinglish Resume Normalization:** Automatically translates Hinglish terms such as `"fresher"` to `"entry level"`, `"passout"` to `"graduate"`, and formats `"b tech"` and `"dot net"`.
- **Indian PII Redaction:** Protects candidate privacy by redacting `+91` phone numbers and 12-digit Aadhaar number patterns.
- **Offline & Low-Bandwidth Resilience:** Runs fully locally to work seamlessly in low-bandwidth environments without dependency on expensive cloud APIs.

