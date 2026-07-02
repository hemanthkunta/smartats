"""
Skill taxonomy + synonym map.
This is what makes the matcher 'smart' instead of doing dumb exact string matching.
e.g. JD says "JS", resume says "JavaScript" -> should match.
JD says "ML", resume says "Machine Learning" -> should match.
"""

# Canonical skill -> list of synonyms/aliases that should be treated as the same skill
SKILL_SYNONYMS = {
    "javascript": ["js", "javascript", "es6", "ecmascript"],
    "typescript": ["ts", "typescript"],
    "python": ["python", "py"],
    "java": ["java", "j2ee", "jee"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp", "dotnet", ".net"],
    "machine learning": ["ml", "machine learning"],
    "deep learning": ["dl", "deep learning"],
    "natural language processing": ["nlp", "natural language processing"],
    "artificial intelligence": ["ai", "artificial intelligence"],
    "react": ["react", "reactjs", "react.js"],
    "node": ["node", "nodejs", "node.js"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "tsql", "plsql"],
    "nosql": ["nosql", "mongodb", "mongo", "dynamodb", "cassandra"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker", "containerization", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab", "version control"],
    "rest api": ["rest", "rest api", "restful", "restful api"],
    "graphql": ["graphql"],
    "data analysis": ["data analysis", "data analytics"],
    "data science": ["data science"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "sass", "scss"],
    "agile": ["agile", "scrum", "kanban"],
    "project management": ["project management", "pm", "pmp"],
    "communication": ["communication", "communication skills"],
    "leadership": ["leadership", "team lead", "led a team", "managed a team"],
    "linux": ["linux", "unix"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "excel": ["excel", "microsoft excel", "ms excel"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "spring boot": ["spring boot", "spring", "springboot"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi", "fast api"],
    "product management": ["product management", "pm", "pmp", "product roadmaps", "jira", "user research", "product discovery", "product strategy"],
    "cybersecurity": ["cybersecurity", "security", "penetration testing", "vulnerability scanning", "vulnerability management", "owasp", "network security", "iam"],
    "mobile development": ["mobile development", "swift", "kotlin", "java", "react native", "mobile ui", "app store", "google play"],
}

from engine.india_taxonomy import INDIA_SKILL_ADDITIONS
SKILL_SYNONYMS.update(INDIA_SKILL_ADDITIONS)

# Build a reverse lookup: alias (lowercase) -> canonical skill name
ALIAS_TO_CANONICAL = {}
for canonical, aliases in SKILL_SYNONYMS.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical

# Common section headers we use to slice a resume into structured chunks
SECTION_HEADERS = {
    "experience": ["experience", "work experience", "employment history", "professional experience"],
    "education": ["education", "academic background", "qualifications"],
    "skills": ["skills", "technical skills", "core competencies", "key skills"],
    "projects": ["projects", "personal projects", "academic projects"],
    "certifications": ["certifications", "certificates", "licenses"],
    "summary": ["summary", "objective", "profile", "about me"],
}

DEGREE_LEVELS = {
    "phd": 5, "doctorate": 5,
    "master": 4, "msc": 4, "m.tech": 4, "mba": 4, "ms": 4, "m.s": 4,
    "bachelor": 3, "bsc": 3, "b.tech": 3, "be": 3, "b.e": 3, "bs": 3, "b.s": 3,
    "diploma": 2,
    "high school": 1, "secondary": 1,
}
