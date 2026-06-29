import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL       = "gemini-1.5-flash"
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

# ─────────────────────────────────────────────
# BharatScore Weights (must sum to 1.0)
# ─────────────────────────────────────────────
WEIGHTS = {
    "semantic_fit":     0.35,   # JD <-> resume semantic alignment
    "career_velocity":  0.20,   # career growth speed
    "skill_recency":    0.20,   # how current are the skills
    "signal_strength":  0.15,   # GitHub / OSS / certifications / hackathons
    "role_alignment":   0.10,   # career arc → target role
}

# ─────────────────────────────────────────────
# India-First Institution Tiers
# (used for understanding, NOT for penalizing)
# ─────────────────────────────────────────────
PREMIER_KEYWORDS   = ["IIT", "IISc", "BITS Pilani"]
STRONG_KEYWORDS    = ["NIT", "IIIT", "BITS", "DAIICT", "TIET", "Thapar", "VJTI", "COEP", "PSG College", "SVNIT", "MNNIT"]
TIER2_KEYWORDS     = ["VIT", "Manipal", "Amrita", "SRM", "Symbiosis", "PEC", "NSIT"]
# Everyone else: good candidate still — we rely on skills, not tags

# ─────────────────────────────────────────────
# Skill Recency Decay (years → weight)
# ─────────────────────────────────────────────
RECENCY_WEIGHTS = {
    0: 1.00,  # acquired this year
    1: 0.90,
    2: 0.78,
    3: 0.65,
    4: 0.52,
    5: 0.40,
}
RECENCY_FLOOR = 0.35   # for skills older than 5 years

# ─────────────────────────────────────────────
# Signal Strength Sub-weights
# ─────────────────────────────────────────────
GITHUB_WEIGHT        = 0.35
OSS_WEIGHT           = 0.25
CERTIFICATION_WEIGHT = 0.25
HACKATHON_WEIGHT     = 0.15

GITHUB_MAX_COMMITS   = 400   # normalisation ceiling
OSS_MAX_CONTRIBS     = 20    # normalisation ceiling
CERT_MAX_COUNT       = 5     # normalisation ceiling

# ─────────────────────────────────────────────
# Hidden Gem thresholds
# ─────────────────────────────────────────────
HIDDEN_GEM_COMMIT_THRESHOLD = 200   # commits/year
HIDDEN_GEM_CERT_THRESHOLD   = 2     # certifications

# ─────────────────────────────────────────────
# FAISS / retrieval
# ─────────────────────────────────────────────
TOP_K_SEMANTIC = 20   # candidates to retrieve before re-ranking

CURRENT_YEAR = 2026
