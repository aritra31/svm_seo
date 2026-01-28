# config.py
from pathlib import Path

HELP_BASE = "https://help.surveymonkey.com"
SEEDS = [
    f"{HELP_BASE}/en/",
    f"{HELP_BASE}/en/surveymonkey/create/",
    f"{HELP_BASE}/en/surveymonkey/send/",
    f"{HELP_BASE}/en/surveymonkey/analyze/",
    f"{HELP_BASE}/en/surveymonkey/billing/",
    f"{HELP_BASE}/en/surveymonkey/getting-started/",
    f"{HELP_BASE}/en/surveymonkey/teams/",
    f"{HELP_BASE}/en/surveymonkey/account/",
    f"{HELP_BASE}/en/surveymonkey/policy/",
    f"{HELP_BASE}/en/surveymonkey/integrations/",
    f"{HELP_BASE}/en/surveymonkey/solutions/"
]


DATA_DIR = Path("data")
CORPUS_DIR = DATA_DIR / "corpus"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Databases
GOLDEN_XLSX_BASE  = DATA_DIR / "SurveyMonkey_Golden_Dataset.xlsx"
GOLDEN_SPLIT_XLSX = DATA_DIR / "golden_split.xlsx"

# ---------------- Strict Evaluation Toggles (default False) ----------------
EVAL_DISABLE_URL_BOOST = False      # True => do NOT boost (expected|acceptable) URLs if present
EVAL_IGNORE_RULE_VARIANTS = False   # True => do NOT include 'variants' from golden set
EVAL_NO_LLM = False                # True => NO LLM reformulation (raw query only)
STRICT_EXPECTED_ONLY = False        # True => ignore 'acceptable_urls' in scoring

REQUEST_SLEEP_RANGE = (0.25, 1.0)
MAX_PAGES = 500   

TOPK_PER_METHOD = 25
FINAL_TOPK = 100
RRF_K = 60

HELP_CENTER_BOOST = 1.1
COMMUNITY_PENALTY = 0.8

USE_RERANKER_DEFAULT = True
RERANK_TOPN = 50
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# CROSS_ENCODER_MODEL = "cross-encoder/nli-deberta-v3-small"


# Reranker / embeddings settings
RERANKER_TYPE = "semantic"   # "none" | "semantic" | "cross"
RERANK_TOPN   = 15           # re-rank top-N from fused list

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CACHE   = CORPUS_DIR / "doc_embeddings.npz"

# Include rule-based variants even in qppgen (recall boost)
ALWAYS_INCLUDE_RULE_VARIANTS = True
RERANKER_TYPE = "semantic"  #fast reranker
