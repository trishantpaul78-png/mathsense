"""
Central configuration for the Adaptive Maths Learning Gaps MVP.

Everything tunable lives here: file paths, Groq settings, BKT priors,
risk thresholds and simulation sizes. Read once at import time.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Make sure the folders exist before anything tries to write to them.
for _folder in (DATA_DIR, MODELS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

DB_PATH = DATA_DIR / "app.db"
SKILLS_PATH = DATA_DIR / "skills.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
RISK_MODEL_PATH = MODELS_DIR / "risk_model.pkl"

# ---------------------------------------------------------------------------
# Groq / LLM (all optional: the app degrades gracefully without a key)
# ---------------------------------------------------------------------------
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()

# Primary model. Change it with GROQ_MODEL in .env; the key's available models are
# listed at https://console.groq.com/docs/models (or GET /openai/v1/models).
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# If the configured model is not available on an account (Groq answers 404
# model_not_found), llm.py silently moves down this list instead of losing AI
# features. Order matters: strongest first.
GROQ_MODEL_FALLBACKS = [
    name.strip()
    for name in os.getenv(
        "GROQ_MODEL_FALLBACKS", "openai/gpt-oss-20b,qwen/qwen3.8-27b"
    ).split(",")
    if name.strip()
]
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "25"))
# Reasoning budget for gpt-oss models: "low" keeps short replies from being
# eaten by hidden reasoning tokens. Set to "" to omit the parameter.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low").strip()
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1200"))

# ---------------------------------------------------------------------------
# Bayesian Knowledge Tracing defaults (per skill overrides below)
# ---------------------------------------------------------------------------
BKT_DEFAULTS = {
    "p_init": 0.45,   # P(student already knows the skill before practice)
    "p_learn": 0.15,  # P(transition to known after one learning opportunity)
    "p_slip": 0.10,   # P(answer wrong | knows it)
    "p_guess": 0.20,  # P(answer right | does not know it)
}

# Later-year topics start lower so the report has visible structure on day one.
BKT_SKILL_OVERRIDES = {
    "linear_systems": {"p_init": 0.38},
    "eigenvalues": {"p_init": 0.32},
    "partial_derivatives": {"p_init": 0.32},
    "applications_derivatives": {"p_init": 0.40},
}

# ---------------------------------------------------------------------------
# Thresholds and thresholds-like knobs
# ---------------------------------------------------------------------------
MASTERY_THRESHOLD = 0.40        # "below mastery" for the risk feature count
ROOT_CAUSE_THRESHOLD = 0.50     # a prerequisite below this is flagged
RISK_THRESHOLD = 0.50           # probability cutoff for "at risk"
RISK_BANDS = ((0.66, "High"), (0.40, "Moderate"), (0.0, "Low"))

# Response-time baseline for the z-score feature (seconds per question).
DEFAULT_RT_MEAN = 18.0
DEFAULT_RT_SD = 9.0

RECENT_WINDOW = 5               # "recent accuracy" window
N_SYNTHETIC_STUDENTS = 200      # simulator.py training pool
ITEMS_PER_SESSION = 20          # simulated session length
DEMO_CLASS_SIZE = 30            # teacher dashboard demo button

# Difficulty label -> IRT b value used when asking Groq for new items.
DIFFICULTY_TO_B = {
    "easy": -1.2,
    "medium": 0.0,
    "hard": 1.2,
}

DISCLAIMER = (
    "This tool identifies learning gaps, not clinical conditions. "
    "Teacher review required."
)
