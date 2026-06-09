"""Central configuration for AgniAI."""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH    = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"
BM25_INDEX_PATH  = INDEX_DIR / "bm25.pkl"

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER   = os.getenv("USE_RERANKER", "0") not in {"0", "false", "False"}

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_URL      = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", f"{OLLAMA_BASE_URL}/api/tags")

DEFAULT_MODEL            = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")
MODEL_MAX_CONTEXT_TOKENS = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_FALLBACK_MODELS",
        "mistral:7b-instruct-q4_K_M,mistral:7b-instruct,mistral:7b-instruct-q4_0,"
        "llama3:8b,llama3:8b-instruct,llama3.1:8b,llama3.2:3b,gemma2:2b",
    ).split(",")
    if m.strip()
]

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_WORDS     = int(os.getenv("CHUNK_WORDS",     "420"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP",   "80"))
CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "12"))

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K                     = int(os.getenv("TOP_K",                     "5"))
RERANK_TOP_K              = int(os.getenv("RERANK_TOP_K",              "4"))
MIN_SCORE                 = float(os.getenv("MIN_SCORE",               "0.20"))
STRICT_MIN_SCORE          = float(os.getenv("STRICT_MIN_SCORE",        "0.55"))
STRICT_TOP_K              = int(os.getenv("STRICT_TOP_K",              "4"))
LOW_RETRIEVAL_CONFIDENCE  = float(os.getenv("LOW_RETRIEVAL_CONFIDENCE",  "0.35"))
HIGH_RETRIEVAL_CONFIDENCE = float(os.getenv("HIGH_RETRIEVAL_CONFIDENCE", "0.60"))
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.88"))
DOMAIN_BOOST_MAX = float(os.getenv("DOMAIN_BOOST_MAX", "1.20"))
ANSWER_GROUNDING_OVERLAP_THRESHOLD = float(
    os.getenv("ANSWER_GROUNDING_OVERLAP_THRESHOLD", "0.75")
)

DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.60"))
BM25_WEIGHT  = float(os.getenv("BM25_WEIGHT",  "0.40"))
USE_HYBRID   = os.getenv("USE_HYBRID", "1") not in {"0", "false", "False"}

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "20"))

# ── Network and cache ──────────────────────────────────────────────────────
REQUEST_TIMEOUT          = int(os.getenv("REQUEST_TIMEOUT", "120"))
# Backward-compatible rename: prefer API_FIRST_TOKEN_TIMEOUT, fall back to legacy FIRST_TOKEN_TIMEOUT.
API_FIRST_TOKEN_TIMEOUT  = int(os.getenv("API_FIRST_TOKEN_TIMEOUT", os.getenv("FIRST_TOKEN_TIMEOUT", "30")))
STREAM_TIMEOUT           = int(os.getenv("STREAM_TIMEOUT", "300"))

RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "3600"))
RESPONSE_CACHE_TTL  = int(os.getenv("RESPONSE_CACHE_TTL",  "86400"))
EMBED_CACHE_TTL     = int(os.getenv("EMBED_CACHE_TTL",     "86400"))
MAX_CACHE_ENTRIES   = int(os.getenv("MAX_CACHE_ENTRIES",   "2048"))

SESSION_HEADER  = os.getenv("SESSION_HEADER", "X-Session-Id")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# ── API Security ───────────────────────────────────────────────────────────
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")

# ── Context char budgets ───────────────────────────────────────────────────
MAX_CONTEXT_CHARS = {
    "short":     int(os.getenv("MAX_CONTEXT_CHARS_SHORT",     "1000")),
    "elaborate": int(os.getenv("MAX_CONTEXT_CHARS_ELABORATE", "1800")),
    "detail":    int(os.getenv("MAX_CONTEXT_CHARS_DETAIL",    "2800")),
}
MAX_CONTEXT_CHARS_DEFAULT = int(os.getenv("MAX_CONTEXT_CHARS_DEFAULT", "1800"))

# ── Token budgets for completion ───────────────────────────────────────────
MAX_TOKENS_STYLE = {
    "short":     int(os.getenv("MAX_TOKENS_SHORT",     "250")),
    "elaborate": int(os.getenv("MAX_TOKENS_ELABORATE", "700")),
    "detail":    int(os.getenv("MAX_TOKENS_DETAIL",    "1100")),
}
MAX_TOKENS_DEFAULT  = int(os.getenv("MAX_TOKENS_DEFAULT",  "600"))
TOKEN_SAFETY_BUFFER = int(os.getenv("TOKEN_SAFETY_BUFFER", "100"))
STYLE_MIN_WORDS = {
    "short": int(os.getenv("STYLE_MIN_WORDS_SHORT", "50")),
    "elaborate": int(os.getenv("STYLE_MIN_WORDS_ELABORATE", "120")),
    "detail": int(os.getenv("STYLE_MIN_WORDS_DETAIL", "250")),
}

# =============================================================================
# INTENT CLASSIFICATION PHRASE LISTS
# =============================================================================

# ── Negation signals used inside classify_intent ───────────────────────────
_NEGATION_SIGNALS = (
    "not", "no ", "can't", "cannot", "don't", "didn't", "failed",
    "rejected", "disqualified", "ineligible", "without", "unable",
    "if i don't", "if i fail", "even if", "despite",
    "won't", "wouldn't", "couldn't", "shouldn't",
    "over age", "overage", "under age", "underage",
    "too old", "too young", "too short", "too light", "too heavy",
    "someone who", "person who", "people who",
)

_NEGATION_DOMAIN_TERMS = (
    "join", "apply", "eligible", "qualify", "pass", "medical",
    "physical", "age", "height", "document", "certificate",
    "salary", "training", "selection", "exam", "rally", "ncc",
    "weight", "chest", "run", "beam", "agniveer", "agnipath",
)

# ─── Fuzzy intent vocabulary ────────────────────────────────────────────────
FUZZY_VOCAB: dict[str, str] = {
    "salar": "salary",
    "slary": "salary",
    "salry": "salary",
    "slaary": "salary",
    "salay": "salary",
    "sallary": "salary",
    "pakage": "package",
    "pacakge": "package",
    "packge": "package",
    "stipand": "stipend",
    "stipend": "stipend",
    "inhand": "in hand",
    "in-hand": "in hand",
    "elgiible": "eligible",
    "eligble": "eligible",
    "eligiblity": "eligibility",
    "eligibilty": "eligibility",
    "eigibility": "eligibility",
    "qualifcation": "qualification",
    "qualifiction": "qualification",
    "qulification": "qualification",
    "phyiscal": "physical",
    "physial": "physical",
    "fitnees": "fitness",
    "fittness": "fitness",
    "heigth": "height",
    "hieght": "height",
    "wieght": "weight",
    "trainng": "training",
    "trainnig": "training",
    "trianing": "training",
    "documnet": "document",
    "documant": "document",
    "certifcate": "certificate",
    "certifiate": "certificate",
    "agniver": "agniveer",
    "agniverr": "agniveer",
    "agnipath": "agnipath",
    "agnieer": "agniveer",
    "medicla": "medical",
    "mediacl": "medical",
    "mdeical": "medical",
    "writen": "written",
    "examn": "exam",
    "exma": "exam",
    "sevanidhi": "seva nidhi",
    "sevanidh": "seva nidhi",
    "sevnidhi": "seva nidhi",
    "corpous": "corpus",
    "corupus": "corpus",
}


def _fuzzy_normalize_query(query: str) -> str:
    """
    Replace known misspellings with canonical terms before classification
    and before RAG retrieval query expansion.
    """
    words = query.split()
    corrected = []
    for word in words:
        suffix = ""
        while word and word[-1] in "?.,!:;":
            suffix = word[-1] + suffix
            word = word[:-1]
        canonical = FUZZY_VOCAB.get(word.lower())
        corrected.append((canonical or word) + suffix)
    return " ".join(corrected)

# ── Greetings — exact match ────────────────────────────────────────────────
GREETING_PHRASES = {
    "hi", "hello", "hey", "hii", "hiii", "heyy", "heya",
    "hola", "howdy", "yo", "sup", "wassup", "whatsup",
    "bye", "goodbye", "good bye", "see you", "see ya",
    "take care", "good night", "goodnight", "good day",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thank u", "thankyou",
    "ty", "thx", "thanks a lot", "thank you so much",
    "many thanks", "dhanyawad", "shukriya", "meherbani",
    "ok", "okay", "ok thanks", "okay thanks", "alright",
    "got it", "understood", "sure", "noted", "cool",
    "great", "nice", "awesome", "wonderful", "amazing",
    "fantastic", "excellent", "perfect", "good",
    "well done", "good job", "nice work", "keep it up", "carry on",
    "jay hind", "jai hind", "vande mataram",
    "bharat mata ki jai", "jai bharat", "jai jawan",
    "mera bharat mahan", "tiranga", "salute",
    "jai mata di", "har har mahadev", "sat sri akal",
    "jai jawan jai kisan", "jai kisan", "jai vigyan",
    "hindustan zindabad", "bharat zindabad", "india zindabad",
    "inquilab zindabad", "jai rajputana", "jai mahakal",
    "waheguru ji ka khalsa", "bum bum bhole",
    "bol bajrang bali ki jai", "durga mata ki jai",
    "indian army zindabad", "army zindabad",
    "namaste", "namaskar", "pranam", "namasté",
    "jai shree ram", "jai shri ram", "ram ram", "jai siya ram",
    "yes sir", "sir", "ma'am", "mam",
    "attention", "at ease", "dismissed",
    "roger", "roger that", "copy that", "wilco",
    "over", "out", "fall in", "stand easy",
    "you are helpful", "youre helpful", "you're great",
    "you are great", "you are good", "youre good",
    "nice bot", "good bot", "helpful bot", "great bot",
}

# ── Small talk — partial / substring match ─────────────────────────────────
SMALL_TALK_PHRASES = (
    "how are you", "how r you", "how are u", "how r u",
    "how do you do", "how is it going", "how's it going",
    "hows it going", "how are things", "how are you doing",
    "are you okay", "you okay", "u okay",
    "what's up", "whats up", "what is up",
    "who are you", "what are you", "what is your name",
    "whats your name", "what's your name", "your name",
    "tell me about yourself", "introduce yourself",
    "what do you do", "what can you do",
    "what can you help", "what can you help me with",
    "how can you help", "how can you help me",
    "what are you capable of",
    "are you a bot", "are you an ai", "are you human",
    "are you real", "who made you", "who created you",
    "who built you", "who developed you",
    "are you chatgpt", "are you gpt",
    "i am happy", "i am sad", "i feel good", "i feel bad",
    "i am bored", "i am tired", "i am stressed", "i am confused",
    "i need help", "help me", "please help", "help",
    "nice to meet you", "nice to meet u",
    "pleased to meet you", "great to meet you",
    "good to meet you", "glad to meet you",
    "you are amazing", "you are wonderful", "you are awesome",
    "i like you", "i love you", "i love this",
    "this is great", "this is helpful",
    "very helpful", "so helpful", "great help", "big help",
    "you there", "are you there", "you available",
    "are you available", "are you online", "u there",
    "hello there", "anyone there", "is anyone there",
    "motivate me", "give me motivation", "i need motivation",
    "inspire me", "give me inspiration", "encourage me", "cheer me up",
    "have a good day", "have a nice day", "have a great day",
    "have a wonderful day", "enjoy your day",
    "take care of yourself", "stay safe", "stay healthy",
    "all the best", "best of luck", "good luck",
    "wish me luck", "fingers crossed",
)

# ── Patriotic / army pride — partial match ─────────────────────────────────
PATRIOTIC_PHRASES = (
    "jay hind", "jai hind", "vande mataram",
    "bharat mata ki jai", "jai bharat",
    "inquilab zindabad", "jai jawan jai kisan",
    "jai jawan", "jai kisan", "jai vigyan",
    "mera bharat mahan", "hindustan zindabad",
    "bharat zindabad", "india zindabad",
    "shaurya", "veerta", "parakram", "balidan",
    "shaheed", "sainik", "sena", "fauj", "fauji",
    "desh seva", "rashtra seva", "desh bhakti", "deshbhakti",
    "watan", "tiranga", "tricolor", "national flag",
    "republic day", "independence day", "army day",
    "vijay diwas", "kargil vijay diwas",
    "jai mata di", "durga mata ki jai", "har har mahadev",
    "bum bum bhole", "sat sri akal", "waheguru ji ka khalsa",
    "jai rajputana", "rajputana rifles", "jai mahakal",
    "bol bajrang bali ki jai",
    "indian army zindabad", "army zindabad",
    "proud to be indian", "proud of indian army",
    "salute to army", "salute to soldiers",
    "respect the army", "army is great",
    "soldiers are heroes", "our army is the best",
    "indian army is best", "love indian army",
    "support our troops", "army rocks", "army is life",
)

# ── Date/Time queries — hardcoded response ─────────────────────────────────
# Only include phrases that unambiguously ask for the current date/time,
# not recruitment dates (e.g., "exam date", "rally date").
DATE_QUERY_PHRASES = (
    "what is today's date",
    "what's today's date",
    "current date",
    "today's date",
    "date today",
    "what day is it",
    "what day",
    "today date",
    "tell me the date",
    "date please",
    "today",
    "what is the time",
    "what's the time",
    "current time",
    "what time is it",
    "time now",
)

# ── Agniveer aspirant casual talk ──────────────────────────────────────────
AGNIVEER_CASUAL_PHRASES = (
    "i want to join agniveer",
    "i want to become agniveer",
    "i am an agniveer aspirant",
    "i am preparing for agniveer",
    "i am studying for agniveer",
    "agniveer is my dream",
    "my dream is agniveer",
    "i will become agniveer",
    "i want to serve my country",
    "i want to join army",
    "i want to be a soldier",
    "i want to serve india",
    "i want to join indian army",
    "i love agniveer",
    "agniveer zindabad",
    "agniveer rocks",
    "wish me luck",
    "pray for me",
    "i am nervous",
    "i am excited",
    "i am scared",
    "i am worried",
    "will i pass",
    "can i do it",
    "i will do it",
    "i can do it",
    "i will try my best",
    "i am ready",
    "i am going to give my best",
    "motivate me",
    "give me motivation",
    "i need motivation",
    "inspire me",
    "i feel demotivated",
    "i failed the exam",
    "i failed the rally",
    "i failed agniveer",
    "i got rejected",
    "i did not pass",
    "i did not qualify",
    "better luck next time",
    "i will try again",
    "never give up",
    "agniveer life",
    "life of an agniveer",
    "agniveer is tough",
    "agniveer is hard",
    "agniveer is good",
    "agniveer is great",
    "is agniveer worth it",
    "agniveer worth it",
    "should i join agniveer",
    "thinking of joining agniveer",
    "planning to join agniveer",
    "considering agniveer",
)

# ── Training process questions → RAG ──────────────────────────────────────
TRAINING_PROCESS_PHRASES = (
    "training process",
    "what is training",
    "tell me about training",
    "explain training",
    "how is the training",
    "what is the training",
    "training like",
    "training duration",
    "how long is training",
    "how long does training",
    "training programme",
    "training program",
    "what happens in training",
    "what happens during training",
    "training period",
    "military training",
    "basic training",
    "what do they train",
    "what will i learn in training",
    "skills taught",
    "training schedule",
    "training syllabus",
    "training curriculum",
    "how long is the training",
    "training for how long",
    "weeks of training",
    "months of training",
    "training centre",
    "training center",
    "regimental training",
    "pass out parade",
    "passing out parade",
    "attestation ceremony",
    "bmt",
    "amt",
    "basic military training",
    "advanced military training",
)

# ── Joining / selection process → RAG ─────────────────────────────────────
PROCESS_PHRASES = (
    "how to join",
    "how do i join",
    "how can i join",
    "how to apply",
    "how do i apply",
    "how can i apply",
    "selection process",
    "recruitment process",
    "steps to join",
    "what are the steps",
    "joining process",
    "application process",
    "registration process",
    "how is the selection",
    "how does selection work",
    "what is the procedure",
    "full process",
    "complete process",
    "recruitment timeline",
    "recruitment schedule",
    "recruitment calendar",
    "when does recruitment",
    "when will recruitment",
    "recruitment date",
    "notification release",
    "official notification",
)

# ── Word-boundary regex for short/ambiguous RAG trigger terms ─────────────
# Used as step 7c in classify_intent to catch salary/pay/earn synonyms and
# timeline queries that don't appear verbatim in DOMAIN_TERMS entries.
# Using \b avoids false matches like "display" triggering "pay".

_WORD_BOUNDARY_RAG_TERMS = re.compile(
    r"\b(?:"
    # Salary / pay synonyms
    r"pay|earn|earning|earnings|income|emolument|wages|remuneration|"
    r"disburse|take.?home|"
    # Notification / timeline
    r"notification|vacancy|vacancies|"
    # Marital status (common in eligibility questions)
    r"married|marriage|unmarried|marital|spouse|"
    # Application / registration (word-boundary safe — "apply" won't match "appliance")
    r"apply|register|registration|"
    # Physical measurements
    r"weight|chest|height|"
    # Academic / eligibility synonyms
    r"percentage|marks|aggregate|"
    # Document synonyms
    r"document|certificate"
    r")\b",
    re.IGNORECASE,
)

# ── Domain terms — RAG trigger ─────────────────────────────────────────────
# NOTE: ordering matters for readability only; all are checked via "any()"
DOMAIN_TERMS = (
    # ── Age / eligibility ──────────────────────────────────────────────────
    "age limit",
    "eligibility",
    "eligible",
    "qualify",
    "age criteria",
    "minimum age",
    "maximum age",
    "how old",
    # ── Education / qualification ──────────────────────────────────────────
    "qualification",
    "educational qualification",
    "education",
    "class 10",
    "class 12",
    "class 8",
    "10th",
    "12th",
    "8th",
    "matric",
    "minimum marks",
    "percentage required",
    "pass percentage",
    "marks required",
    "marksheet",
    "iti",
    "diploma",
    "intermediate",
    "pcm",
    "physics maths",
    # ── Salary / pay / earnings ────────────────────────────────────────────
    "salary",
    "pay scale",
    "stipend",
    "in hand",
    "in-hand",
    "gross salary",
    "monthly salary",
    "annual salary",
    "per month",
    "per year",
    "how much earn",
    "how much paid",
    "how much do",
    "how much will",
    "how much is",
    "total earning",
    "total salary",
    "total pay",
    "monthly pay",
    "annual pay",
    "monthly income",
    "income",
    "compensation",
    "emolument",
    "allowance",
    "deduction",
    "corpus",
    "package",
    "customised package",
    "pay structure",
    "salary structure",
    "salary breakdown",
    "pay breakdown",
    # ── Seva Nidhi / exit package ──────────────────────────────────────────
    "seva nidhi",
    "exit package",
    "lump sum",
    "11.71",
    "10.04",
    "corpus fund",
    # ── Insurance / compensation ───────────────────────────────────────────
    "insurance",
    "death compensation",
    "disability",
    "ex gratia",
    "48 lakh",
    "compensation",
    "gratuity",
    "pension",
    "family pension",
    # ── Benefits / leave ──────────────────────────────────────────────────
    "benefit",
    "leave",
    "annual leave",
    "sick leave",
    "canteen",
    "csd",
    "income tax",
    "tax exempt",
    # ── Timeline / dates / notification ───────────────────────────────────
    "notification",
    "official notification",
    "timeline",
    "recruitment timeline",
    "schedule",
    "calendar",
    "exam date",
    "rally date",
    "joining date",
    "reporting date",
    "when to apply",
    "last date",
    "apply date",
    "registration date",
    "application date",
    "start date",
    "vacancy",
    "vacancies",
    "when is the exam",
    "when is the rally",
    "when is the notification",
    "when does the",
    "which month",
    "what date",
    # ── Physical / PFT / medical ───────────────────────────────────────────
    "pft",
    "physical fitness test",
    "physical test",
    "fitness test",
    "medical",
    "height requirement",
    "chest measurement",
    "weight requirement",
    "1.6 km run",
    "beam",
    "long jump",
    "pull up",
    "eyesight",
    "dental",
    "hearing",
    "colour vision",
    "color vision",
    "vision",
    "ditch jump",
    "zig zag",
    "pmt",
    "physical measurement",
    # ── Exam / CEE ────────────────────────────────────────────────────────
    "written exam",
    "cee",
    "common entrance",
    "syllabus",
    "exam pattern",
    "negative marking",
    "typing test",
    "admit card",
    "merit list",
    "cut off",
    "cutoff",
    "exam fee",
    "examination fee",
    "250",
    "online exam",
    "computer based",
    "cbt",
    "question paper",
    "marking scheme",
    "exam centre",
    "exam center",
    # ── Documents ─────────────────────────────────────────────────────────
    "document",
    "domicile",
    "aadhaar",
    "character certificate",
    "caste certificate",
    "ncc certificate",
    "document required",
    "unmarried certificate",
    "relation certificate",
    # ── Bonus marks ───────────────────────────────────────────────────────
    "bonus mark",
    "ncc",
    "sports bonus",
    "sos",
    "soex",
    # ── Training ──────────────────────────────────────────────────────────
    "training",
    "regimental centre",
    "induction",
    "drill",
    "weapon training",
    "firing",
    "combat training",
    "fieldcraft",
    "field craft",
    "map reading",
    "physical conditioning",
    "regiment",
    "battalion",
    # ── Service / posting ─────────────────────────────────────────────────
    "posting",
    "deployment",
    "pass out",
    "passout",
    "allotment",
    "completion certificate",
    "rally",
    "agnipath",
    # ── Retention / permanent cadre ───────────────────────────────────────
    "retention",
    "permanent",
    "regular cadre",
    "25 percent",
    "25%",
    "permanent selection",
    # ── Post-service ──────────────────────────────────────────────────────
    "ex servicemen",
    "esm",
    "capf",
    "reservation",
    "government job",
    "after agniveer",
    "after service",
    "after 4 years",
    "exit",
    "leaving service",
    "service duration",
    "total service",
    "how many years",
    "skill certificate",
    "nios",
    "class 12 equivalent",
    # ── Application / registration ────────────────────────────────────────
    "registration",
    "apply online",
    "online application",
    "join indian army",
    "jia website",
    "application fee",
    "exam fee",
    "operation sindoor",
    "discharge",
    "release from service",
    "married during service",
    # 2026 KB additions
    "category x",
    "category y",
    "category z",
    "disability compensation",
    "44 lakh",
    "natural death",
    "service death",
    "operational death",
    "martyr",
    "sindoor",
    "drone",
    "combat deployment",
    "retention rate",
    "75 percent retention",
    "jaisalmer conference",
    "policy revision",
    "scheme revision",
    "bmt",
    "amt",
    "basic military training",
    "advanced military training",
    "on the job training",
    "ojt",
    "passing out parade",
    "attestation",
    "week 24",
    "week 31",
    "chest number",
    "piq",
    "shaurya bharat",
    "psychological test",
    "mental test",
    "dipr",
    "stress test",
    "national trade certificate",
    "ntc",
    "ignou",
    "degree program",
    "bsf reservation",
    "cisf reservation",
    "rpf reservation",
    "ex agniveer",
    "ex-agniveer wing",
    "income tax exempt",
    "tax free",
    "high altitude allowance",
    "ltc",
    "leave travel",
    "ration allowance",
    "dress allowance",
    "risk hardship",
    "no hra",
    "no da",
    "no msp",
    "no pension",
    "no gratuity",
    "digilocker",
    "aadhaar verification",
    "e-recruitex",
    "anti touting",
    "tout",
    "helpline",
    "9289914527",
)

# ── Reasoning / calculation terms ─────────────────────────────────────────
REASONING_TERMS = (
    "calculate",
    "total",
    "sum",
    "overall",
    "aggregate",
    "combined",
    "after 4 years",
    "over 4 years",
    "for 4 years",
    "how long",
    "how many weeks",
    "how many months",
    "how many days",
    "duration of",
    "length of",
    "when does",
    "when will",
    "when is",
    "when are",
    "what date",
    "which month",
    "what happens after",
    "what happens during",
    "in total",
    "per year total",
    "4 year total",
)

REASONING_SALARY_TERMS = (
    "salary",
    "pay",
    "service",
    "seva",
    "benefit",
    "nidhi",
    "year",
    "years",
    "earn",
    "income",
    "package",
    "stipend",
    "corpus",
)

# ── Answer-style keywords ──────────────────────────────────────────────────
STYLE_SHORT_KEYWORDS = [
    "in short", "briefly", "brief", "quick answer", "short answer",
    "summarise", "summarize", "tldr", "tl;dr", "in brief",
    "give me short", "one line", "one-line", "give a short",
    "keep it short", "summary", "summarise it", "quick summary",
]

STYLE_DETAIL_KEYWORDS = [
    "in detail", "detailed", "explain in detail", "full detail",
    "comprehensive", "thoroughly", "exhaustive", "step by step",
    "step-by-step", "explain fully", "tell me everything",
    "give me detail", "elaborate in detail", "full explanation",
    "complete explanation", "everything about", "all about",
    "full breakdown", "break it down",
]

STYLE_ELABORATE_KEYWORDS = [
    "elaborate", "explain", "elaborate on", "tell me more",
    "expand on", "describe", "give more", "more info",
    "more detail", "walk me through", "how does", "how do",
]


def _style_kw_match(query_lower: str, keywords: list[str]) -> bool:
    for kw in keywords:
        if " " in kw:
            if kw in query_lower:
                return True
        elif re.search(rf"\b{re.escape(kw)}\b", query_lower):
            return True
    return False


def detect_answer_style(query: str) -> tuple[str, str]:
    q = (query or "").lower()
    if _style_kw_match(q, STYLE_SHORT_KEYWORDS):
        return "short", "short"
    if _style_kw_match(q, STYLE_DETAIL_KEYWORDS):
        return "detail", "detail"
    if _style_kw_match(q, STYLE_ELABORATE_KEYWORDS):
        return "elaborate", "elaborate"
    return "elaborate", "elaborate"

# ── Style output guidance ──────────────────────────────────────────────────
STYLE_OUTPUT_GUIDANCE = {
    "short": (
        "OUTPUT FORMAT — SHORT (strict):\n"
        "Write exactly ONE concise paragraph of 60-90 words.\n"
        "State only the key facts. No elaboration, no repetition.\n"
        "Use ONLY the provided reference information.\n"
        "If the answer is not in the reference, respond exactly with the configured "
        "reference fallback message."
    ),
    "elaborate": (
        "OUTPUT FORMAT — ELABORATE (strict):\n"
        "Write 2 to 3 paragraphs totalling 150-250 words.\n"
        "First paragraph: core facts. Second paragraph: supporting details or context. "
        "Third paragraph (optional): practical implications or exceptions.\n"
        "Use ONLY the provided reference information. "
        "No bullet points unless listing more than 4 items.\n"
        "If the answer is not in the reference, respond exactly with the configured "
        "reference fallback message."
    ),
    "detail": (
        "OUTPUT FORMAT — DETAIL (strict):\n"
        "Write 3 to 5 paragraphs totalling 300-450 words.\n"
        "Cover: (1) core facts, (2) specific requirements or conditions, "
        "(3) exceptions or special cases, (4) process or procedure if applicable, "
        "(5) important notes or warnings.\n"
        "Use ONLY the provided reference information. "
        "Be thorough but do not invent information.\n"
        "If the answer is not in the reference, respond exactly with the configured "
        "reference fallback message."
    ),
}

STYLE_POINT_TOKEN_BUDGET = {
    "short":     int(os.getenv("STYLE_SHORT_POINT_TOKENS",     "0")),
    "elaborate": int(os.getenv("STYLE_ELABORATE_POINT_TOKENS", "80")),
    "detail":    int(os.getenv("STYLE_DETAIL_POINT_TOKENS",    "150")),
}

# ── RAG system prompts ─────────────────────────────────────────────────────
_PARAGRAPH_RULES = (
    "RULES (follow strictly for every response):\n"
    "1. Use ONLY the provided context.\n"
    "2. Do NOT use prior knowledge.\n"
    "3. Do NOT add explanations, tips, assumptions, or examples not in the context.\n"
    "4. Do NOT infer missing information.\n"
    "5. If the answer is incomplete or missing, respond exactly with the configured "
    "reference fallback message.\n"
    "6. Do NOT repeat the question.\n"
    "7. Do NOT include generic filler sentences.\n"
    "8. Write in clear structured paragraphs only. Avoid bullet overload.\n"
    "9. Deliver all key facts first, compress the remainder, never cut mid-sentence.\n"
    "10. Every sentence must be grammatically complete and end cleanly."
)
STRICT_RAG_PROMPT = (
    "You are a precise, production-grade question-answering system for the "
    "Agniveer / Agnipath training process of the Indian Armed Forces.\n\n"
    "SOURCE RULE — ABSOLUTE:\n"
    "The reference information provided is your ONLY source. "
    "Do not use prior training knowledge. Do not guess. Do not infer numbers "
    "that are not explicitly written. Do not blend reference facts with memory.\n\n"
    "HANDLING CONFLICTING SOURCES:\n"
    "The knowledge base contains documents from different dates. If two retrieved "
    "chunks give different values for the same fact (e.g., Year 3 in-hand salary "
    "appears as ₹25,550 in the official Army e-book and ₹25,580 in a 2026 analysis "
    "document), you MUST report both values and note the discrepancy:\n"
    "'Note: the reference sources do not agree — Source A states [X], Source B "
    "states [Y]. Please verify with joinindianarmy.nic.in.'\n"
    "Never silently pick one value and discard the other.\n\n"
    "NUMBER ACCURACY RULES:\n"
    "1. Copy every number from the reference exactly as written.\n"
    "2. If the reference contains a table, read each row carefully. "
    "   Do not mix values from different rows or different years.\n"
    "3. If Gross, Deduction, and In-hand are all stated, verify: "
    "   Gross minus Deduction must equal In-hand. "
    "   If not, report the mismatch rather than guessing.\n"
    "4. Never perform arithmetic beyond simple subtraction to verify a stated value.\n"
    "5. Never write a bare number. Always give it full context: "
    "   'Rs 21,000 per month (in-hand, Year 1)' — not just '21000'.\n"
    "6. If a number's context is ambiguous (which year? which category?), label it "
    "   explicitly or state the ambiguity rather than presenting it without context.\n\n"
    "WHEN THE ANSWER IS NOT IN THE REFERENCE:\n"
    "Respond with exactly:\n"
    "\"I'm sorry, this information is not available in my reference. "
    "Please check joinindianarmy.nic.in or contact your nearest "
    "Army Recruitment Office.\"\n\n"
    "FORMAT:\n"
    "Do not repeat the question. Do not add filler phrases. "
    "Write in clear, complete paragraphs. "
    "Use a numbered list only when the content is genuinely sequential or parallel. "
    "Deliver key facts first. Never cut a sentence mid-thought."
)

STRICT_RAG_PROMPT_COMPUTE = (
    "You are a precise, production-grade question-answering system for the "
    "Agniveer / Agnipath training process of the Indian Armed Forces. "
    "This question requires reading and computing from a table or structured data.\n\n"
    "SOURCE RULE — ABSOLUTE:\n"
    "Use ONLY the provided reference. Do not use prior knowledge. "
    "Do not guess missing values. Do not extrapolate.\n\n"
    "HANDLING CONFLICTING SOURCES:\n"

    "If two retrieved chunks give different values for the same figure "
    "(for example, Year 3 in-hand salary is ₹25,550 in the official Army e-book "
    "and ₹25,580 in a 2026 lifecycle report), do NOT silently pick one. "
    "Report both, show which source says what, and note the discrepancy. "
    "Use the official Army e-book figure as primary but flag the conflict.\n\n"

    "COMPUTATION RULES:\n"
    "1. Before computing, identify every relevant row and figure in the reference.\n"
    "2. Only add, subtract, or multiply values explicitly present in the reference.\n"
    "3. Show your working transparently when summing across years or categories:\n"
    "   'Year 1: Rs X + Year 2: Rs Y + Year 3: Rs Z + Year 4: Rs W = Rs Total'\n"
    "   This lets the user verify your arithmetic.\n"

    "4. If the reference contains a numerical inconsistency "
    "   (Gross minus Deduction does not equal stated In-hand), "
    "   report it: 'Note: these figures do not balance — [show mismatch]. "
    "   Please verify with joinindianarmy.nic.in.'\n"
    "5. Label every number with its unit and context. Never write a bare number.\n"
    "6. If a required value is missing from the reference, say so. "
    "   Do not estimate. Do not fill gaps with training memory.\n\n"
    "FORMAT:\n"
    "Do not repeat the question. Show calculations step by step. "
    "Conclude with the final answer in plain language."
)

CONDITIONAL_RAG_PROMPT = (
    "You are a document-based QA system for the Agniveer / Agnipath "
    "training process of the Indian Armed Forces. "
    "The user is asking a conditional or negated question (e.g. what happens if they fail, "
    "or whether someone without a certain qualification can apply). "
    "Use ONLY the provided context. You MAY reason about eligibility from the stated requirements — "
    "for example, if the context states a minimum height of 170cm, you may infer that someone "
    "below 170cm would not meet that requirement. "
    "Do NOT invent facts not present in the context. "
    "Be direct and helpful. Do not repeat the question. "
    "If the context truly contains no relevant information, respond with the configured "
    "reference fallback message."
)

SYSTEM_PROMPT_SHORT     = STRICT_RAG_PROMPT
SYSTEM_PROMPT_ELABORATE = STRICT_RAG_PROMPT
SYSTEM_PROMPT_DETAIL    = STRICT_RAG_PROMPT
SYSTEM_PROMPT           = STRICT_RAG_PROMPT

REFERENCE_FALLBACK = (
    "I'm sorry, this information is not available in my reference. "
    "Please check joinindianarmy.nic.in or contact your nearest "
    "Army Recruitment Office."
)

SOURCE_PRIORITY_PROMPT = (
    "CONTROLLED MULTI-SOURCE RETRIEVAL POLICY:\n"
    "1. The JSON knowledge base chunks provided as reference information are the "
    "primary and authoritative source.\n"
    "2. If relevant information exists in the reference information, use only "
    "that information. Do not override it, correct it, or blend it with prior "
    "model knowledge.\n"
    "3. First identify all relevant reference chunks, then extract the complete "
    "facts needed for the answer. Do not pick isolated values from unrelated "
    "chunks.\n"
    "4. For numerical answers, explain what each number represents. If multiple "
    "numbers appear, label their roles such as base, deduction, gross, or final. "
    "Verify relationships before answering, especially Final = Base - Deduction.\n"
    "5. Never return a number without context. Never invent or assume a number.\n"
    "6. If the reference information is incomplete or missing for the user's "
    "question, respond exactly with the configured reference fallback message.\n"
    "7. After grounding the facts, phrase the answer naturally and simply without "
    "changing the facts."
)

GENERAL_KNOWLEDGE_FALLBACK_PROMPT = (
    "The knowledge base did not return a reliable answer for this question.\n\n"
    "DECISION RULE — follow this order:\n\n"
    "STEP 1 — Is this casual, personal, or emotional?\n"
    "(Signs: sharing feelings, asking for motivation, making small talk, "
    "expressing pride, sadness, nervousness, or excitement.)\n"
    "If YES: respond like a warm, caring human. "
    "Do not say 'I do not have that information'. "
    "No bullet points or structured formatting. "
    "Be brief, genuine, encouraging. Acknowledge what they said before "
    "offering anything else. 1–3 sentences maximum.\n\n"
    "STEP 2 — Is this a factual or subject-based question?\n"
    "(Signs: asking what something is, how it works, who someone is, "
    "when something happened, or why something is the case.)\n"
    "If YES: answer from general knowledge ONLY if you are genuinely confident. "
    "'Confident' means you would stake your reputation on it — not just that "
    "it sounds plausible.\n"
    "Signal general-knowledge answers: 'From what I know...' or 'Generally...'\n\n"
    "STEP 3 — Numbers and specifics:\n"
    "Never state a specific number — salary, age limit, percentage, date, "
    "fee, measurement, count — unless you are completely certain. "
    "If even slightly unsure, leave it out and say:\n"
    "'I do not have the verified figure for that. Please check "
    "joinindianarmy.nic.in or contact your nearest Army Recruitment Office.'\n\n"
    "STEP 4 — If you cannot answer:\n"
    "Say so directly, warmly, and briefly. Do not pad. "
    "'I do not have reliable information on that. For accurate details, "
    "please check joinindianarmy.nic.in or contact your nearest Army Recruitment "
    "Office.' That is a complete, honest, and helpful answer.\n\n"
    "WHAT YOU MUST NEVER DO:\n"
    "- Never say 'Answer not found in the document' — there is no document here.\n"
    "- Never mix partial reference facts with general knowledge guesses.\n"
    "- Never invent a number and present it as fact.\n"
    "- Never give a structured, bullet-pointed reply to a casual personal message.\n"
    "- Never be dismissive of the user's emotion or situation.\n"
)

CHAT_SYSTEM_PROMPT = (
    "You are AgniAI — a knowledgeable, warm, and patriotic AI assistant built to "
    "help young Indians understand and prepare for the Agniveer / Agnipath training "
    "process of the Indian Armed Forces.\n\n"
    "━━ WHO YOU ARE ━━\n"
    "Think of yourself as a senior Agniveer — someone who has been through every "
    "stage of the process and genuinely wants the person in front of you to succeed. "
    "You speak like a real person, not a helpdesk. Warm but not fake. Patriotic "
    "but not preachy. You talk the way a trusted, experienced friend would.\n\n"
    "━━ WHAT YOU KNOW ━━\n"
    "Your knowledge base covers the complete Agniveer journey:\n"
    "Registration → CEE (written exam, ₹250 fee) → Recruitment Rally (PFT, PMT) "
    "→ Medical → Merit List → Regimental Centre → 31-week training (BMT 10wk + "
    "AMT 21wk + OJT 6wk) → Deployment (3.5 years) → Seva Nidhi exit package.\n\n"
    "Key verified facts (use ONLY when reference information confirms them):\n"
    "- Salary: Year 1 ₹30,000 gross / ₹21,000 in-hand; Year 4 ₹40,000 / ₹28,000\n"
    "- Seva Nidhi: ~₹10.04 lakh (ex-interest), ~₹11.71 lakh (with interest)\n"
    "- Life insurance: ₹48 lakh (non-contributory — free, no deduction from salary)\n"
    "- Age: 17.5 to 21–22 years depending on category and notification year\n"
    "- Exam fee: ₹250\n"
    "- Leave: 30 days annual, sick leave as per medical advice\n"
    "- Training: 31 weeks total (attestation Week 24, Passing Out Parade Week 31)\n"
    "- Retention for regular cadre: up to 25% per batch (proposals for more in 2026)\n"
    "- Operation Sindoor (May 2025): ~3,000 Agniveers on western front — first "
    "  major combat deployment\n"
    "- No pension, no gratuity, no ESM status for the 75% who exit\n"
    "- Income tax exempt during service and on Seva Nidhi payout\n"
    "- Only unmarried candidates can enroll; must stay unmarried for 4 years\n\n"
    "━━ HOW TO RESPOND — THREE SITUATIONS ━━\n\n"
    "SITUATION 1 — Reference information is provided in this conversation:\n"
    "Use ONLY those retrieved facts. Do not mix in your own knowledge. "
    "Do not add any number — salary, age, percentage, fee, date — that is not "
    "explicitly written in the reference. If the reference is partial, say so: "
    "'Based on what I have, [answer]. For full details, check joinindianarmy.nic.in.'\n\n"
    "SITUATION 2 — No reference, but you know the answer confidently:\n"
    "Answer from general knowledge only when you are genuinely certain. "
    "Signal it: 'From what I know...' or 'Generally speaking...'. "
    "If even slightly unsure of a specific number or date, say: "
    "'I don't have the verified figure for that — please check "
    "joinindianarmy.nic.in or contact your nearest Army Recruitment Office.'\n\n"
    "SITUATION 3 — Casual conversation, greetings, emotions, patriotic talk:\n"
    "Respond like a real human. Match the energy. Short, warm, genuine. "
    "Someone says Jay Hind — respond with equal pride. "
    "Someone is nervous before their rally — be the senior who has been there. "
    "Someone says hi — just say hi back, naturally. "
    "Someone failed and is sad — acknowledge it first, then offer perspective. "
    "NO bullet points, NO headers, NO structured formatting for casual replies. "
    "One to three sentences maximum.\n\n"
    "━━ READING THE USER'S INTENT ━━\n"
    "Before replying, ask: what does this person actually need right now?\n"
    "- FACT → give accurate information, well organised.\n"
    "- MOTIVATION → warm, brief, real. No lecture.\n"
    "- ACKNOWLEDGEMENT → hear them first, then help.\n"
    "- GREETING → greet back, invite their question.\n"
    "- PRIDE / EXCITEMENT → match that energy.\n\n"
    "━━ NUMBER ACCURACY — NON-NEGOTIABLE ━━\n"
    "Salary, corpus amounts, age limits, height/weight/chest requirements, "
    "mark cutoffs, fees, insurance figures, leave days — users trust these most. "
    "One wrong number makes the whole chatbot untrustworthy. "
    "Rule: if a number did not come from retrieved reference in this conversation, "
    "do not state it. Acknowledge the question, say you need to verify, and direct "
    "them to joinindianarmy.nic.in.\n\n"
    "━━ DATA CONFLICTS IN THE KNOWLEDGE BASE ━━\n"
    "The knowledge base contains documents from different years. When sources "
    "disagree on a number (e.g., Year 3 in-hand salary is ₹25,550 in the official "
    "Army e-book and ₹25,580 in another document), always report both and note "
    "the discrepancy. Direct the user to the official source for verification.\n\n"
    "━━ FORMAT AND TONE ━━\n"
    "- Casual messages → casual tone, short reply, no bullets, no headers.\n"
    "- Factual questions → accurate, organised, paragraphs preferred unless "
    "  listing genuinely parallel items.\n"
    "- Always say 'Agniveer training process' — never 'recruitment scheme'.\n"
    "- Never open with 'Certainly!', 'Great question!', 'Of course!', or "
    "  'Absolutely!' — these sound scripted.\n"
    "- Never say 'Answer not found in the document' to someone making small talk "
    "  or sharing a feeling — that is a system error message, not a human reply.\n"
    "- Never give the same canned greeting every time — vary it naturally.\n"
    "- Short message → short reply. Long factual question → complete answer.\n"
    "- When someone is emotional — respond to the emotion first.\n\n"
    "━━ WHAT YOU NEVER DO ━━\n"
    "- Never invent a number and present it as fact.\n"
    "- Never say 'recruitment scheme' — always 'Agniveer training process'.\n"
    "- Never use bullets for a casual one-line reply.\n"
    "- Never pivot immediately to 'Ask me about eligibility!' after someone "
    "  shares something personal — that feels dismissive.\n"
    "- Never be more than one sentence longer than the question deserves.\n"
)

CHAT_SYSTEM_PROMPT += (
    "\n\nADDITIONAL SMALL-TALK RULE:\n"
    "If the user asks a polite personal question such as 'How are you?', "
    "'How are you doing?', or 'How do you do?', answer politely the way a person "
    "normally would in conversation, for example 'I'm doing well, thank you' or "
    "'I'm good, thanks for asking,' and then continue naturally.\n"
    "Do not reply with robotic lines like 'I am just a machine', "
    "'I do not have feelings', or 'I am only an AI assistant' unless the user is "
    "specifically asking whether you are human or a machine."
)


def style_structure_instruction(style: str) -> str:
    style_key = (style or "").strip().lower()
    guidance  = STYLE_OUTPUT_GUIDANCE.get(style_key, STYLE_OUTPUT_GUIDANCE["elaborate"])
    return f"{SOURCE_PRIORITY_PROMPT}\n\n{_PARAGRAPH_RULES}\n\n{guidance}"


# ── Token estimation utilities ─────────────────────────────────────────────

def estimate_text_tokens(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    raw = max(1, (len(text) + 2) // 3)
    return int(raw * 1.10)


def estimate_message_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        total += 6
        total += estimate_text_tokens(str(message.get("role", "")))
        total += estimate_text_tokens(str(message.get("content", "")))
    return total


def trim_to_complete_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    matches = list(re.finditer(r"[.!?](?:\s|$)", text))
    if not matches:
        if len(text.split()) >= 10:
            return text
        return text
    last_end = matches[-1].end()
    trimmed = text[:last_end].strip()
    if trimmed and len(trimmed) >= len(text) * 0.50:
        return trimmed
    return text


def _is_greeting(q: str) -> bool:
    """Return True for exact-match greetings."""
    return q in GREETING_PHRASES


def _is_small_talk(q: str) -> bool:
    """Return True for substring-based small talk."""
    return any(phrase in q for phrase in SMALL_TALK_PHRASES)


def _is_patriotic(q: str) -> bool:
    """Return True for patriotic or army-pride phrasing."""
    return any(phrase in q for phrase in PATRIOTIC_PHRASES)


def _is_date_query(q: str) -> bool:
    """Return True for date/time queries that should get hardcoded current date/time."""
    for phrase in DATE_QUERY_PHRASES:
        # Use word boundaries to avoid matching phrases like "exam date"
        if re.search(r'\b' + re.escape(phrase) + r'\b', q):
            return True
    return False


def _get_current_date_response() -> str:
    """Return a hardcoded current date/time response."""
    now = datetime.now()
    return now.strftime("Today is %A, %B %d, %Y. Current time is %I:%M %p.")


def _is_agniveer_casual(q: str) -> bool:
    """Return True for Agniveer-aspirant casual talk."""
    return any(phrase in q for phrase in AGNIVEER_CASUAL_PHRASES)


def _is_training_rag(q: str) -> bool:
    """Return True for training-process questions."""
    return any(phrase in q for phrase in TRAINING_PROCESS_PHRASES)


def _is_process_rag(q: str) -> bool:
    """Return True for joining or selection-process questions."""
    return any(phrase in q for phrase in PROCESS_PHRASES)


def _is_domain_rag(q: str) -> bool:
    """Return True for domain-specific Agniveer questions."""
    return any(term in q for term in DOMAIN_TERMS) or bool(_WORD_BOUNDARY_RAG_TERMS.search(q))


def _is_negated_domain_rag(q: str) -> bool:
    """Return True for negated domain questions."""
    return any(sig in q for sig in _NEGATION_SIGNALS) and any(
        term in q for term in _NEGATION_DOMAIN_TERMS
    )


def _is_reasoning_rag(q: str) -> bool:
    """Return True for reasoning-heavy salary or service questions."""
    return any(term in q for term in REASONING_TERMS) and any(
        term in q for term in REASONING_SALARY_TERMS
    )


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================

def classify_intent(query: str) -> str:
    """
    Classify query intent as 'chat', 'rag', or 'reject'.

    Priority order: greeting, small talk, patriotic, Agniveer casual,
    training RAG, process RAG, domain RAG, negated domain RAG,
    reasoning RAG, short-chat fallback, then reject.
    """
    query = _fuzzy_normalize_query(query)
    q = query.strip().lower()
    q = q.replace("!", "").replace("?", "").replace("।", "").replace(",", "").strip()
    tokens = [t for t in q.split() if t]

    if not tokens:
        return "chat"

    checks = (
        (_is_greeting, "chat"),
        (_is_small_talk, "chat"),
        (_is_patriotic, "chat"),
        (_is_agniveer_casual, "chat"),
        (_is_training_rag, "rag"),
        (_is_process_rag, "rag"),
        (_is_domain_rag, "rag"),
        (_is_negated_domain_rag, "rag"),
        (_is_reasoning_rag, "rag"),
    )
    for predicate, result in checks:
        if predicate(q):
            return result

    if len(tokens) <= 3:
        return "chat"
    return "reject"
