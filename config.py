"""Central configuration for AgniAI."""

from __future__ import annotations

import os
import re
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
OLLAMA_URL      = os.getenv("OLLAMA_CHAT_URL", f"{OLLAMA_BASE_URL}/api/chat")
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
MIN_RETRIEVAL_CONFIDENCE  = LOW_RETRIEVAL_CONFIDENCE

DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.60"))
BM25_WEIGHT  = float(os.getenv("BM25_WEIGHT",  "0.40"))
USE_HYBRID   = os.getenv("USE_HYBRID", "1") not in {"0", "false", "False"}

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "10"))

# ── Network and cache ──────────────────────────────────────────────────────
REQUEST_TIMEOUT     = int(os.getenv("REQUEST_TIMEOUT",     "120"))
FIRST_TOKEN_TIMEOUT = int(os.getenv("FIRST_TOKEN_TIMEOUT", "30"))
STREAM_TIMEOUT      = int(os.getenv("STREAM_TIMEOUT",      "300"))

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

# ── Style output guidance ──────────────────────────────────────────────────
STYLE_OUTPUT_GUIDANCE = {
    "short": (
        "OUTPUT FORMAT — SHORT (strict):\n"
        "Write exactly ONE concise paragraph of 60-90 words.\n"
        "State only the key facts. No elaboration, no repetition.\n"
        "Use ONLY the provided reference information.\n"
        "If the answer is not in the reference, respond exactly: "
        "'Answer not found in the document.'"
    ),
    "elaborate": (
        "OUTPUT FORMAT — ELABORATE (strict):\n"
        "Write 2 to 3 paragraphs totalling 150-250 words.\n"
        "First paragraph: core facts. Second paragraph: supporting details or context. "
        "Third paragraph (optional): practical implications or exceptions.\n"
        "Use ONLY the provided reference information. "
        "No bullet points unless listing more than 4 items.\n"
        "If the answer is not in the reference, respond exactly: "
        "'Answer not found in the document.'"
    ),
    "detail": (
        "OUTPUT FORMAT — DETAIL (strict):\n"
        "Write 3 to 5 paragraphs totalling 300-450 words.\n"
        "Cover: (1) core facts, (2) specific requirements or conditions, "
        "(3) exceptions or special cases, (4) process or procedure if applicable, "
        "(5) important notes or warnings.\n"
        "Use ONLY the provided reference information. "
        "Be thorough but do not invent information.\n"
        "If the answer is not in the reference, respond exactly: "
        "'Answer not found in the document.'"
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
    "5. If the answer is incomplete or missing, respond exactly: "
    "'Answer not found in the document.'\n"
    "6. Do NOT repeat the question.\n"
    "7. Do NOT include generic filler sentences.\n"
    "8. Write in clear structured paragraphs only. Avoid bullet overload.\n"
    "9. Deliver all key facts first, compress the remainder, never cut mid-sentence.\n"
    "10. Every sentence must be grammatically complete and end cleanly."
)
STRICT_RAG_PROMPT = (
    "You are a strict document-based QA system for the Agniveer / Agnipath "
    "training process of the Indian Armed Forces. "
    "Use ONLY the provided context. Do NOT use prior knowledge. "
    "Do NOT infer missing information. "
    "Do NOT add explanations, tips, assumptions, or generic filler. "
    "Deliver all key facts first, keep the answer compact, avoid unnecessary repetition. "
    "If the answer is incomplete or missing, respond exactly with: "
    "'Answer not found in the document.' "
    "Write in clear structured paragraphs only. Do not repeat the question."
)

STRICT_RAG_PROMPT_COMPUTE = (
    "You are a strict document-based QA system for the Agniveer / Agnipath "
    "training process of the Indian Armed Forces. "
    "Use ONLY the provided context. You may compute or aggregate values only "
    "from figures explicitly present in the context. "
    "Do NOT use prior knowledge. Do NOT infer missing information. "
    "Do NOT add explanations, tips, assumptions, or generic filler. "
    "Deliver all key facts first, keep the answer compact, avoid unnecessary repetition. "
    "If the answer is incomplete or missing, respond exactly with: "
    "'Answer not found in the document.' "
    "Write in clear structured paragraphs only. Do not repeat the question."
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
    "If the context truly contains no relevant information, respond: "
    "'Answer not found in the document.'"
)

SYSTEM_PROMPT_SHORT     = STRICT_RAG_PROMPT
SYSTEM_PROMPT_ELABORATE = STRICT_RAG_PROMPT
SYSTEM_PROMPT_DETAIL    = STRICT_RAG_PROMPT
SYSTEM_PROMPT           = STRICT_RAG_PROMPT

REFERENCE_FALLBACK = "Answer not found in the document."

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
    "question, respond exactly: 'Answer not found in the document.'\n"
    "7. After grounding the facts, phrase the answer naturally and simply without "
    "changing the facts."
)

GENERAL_KNOWLEDGE_FALLBACK_PROMPT = (
    "The JSON knowledge base did not contain relevant reference information for "
    "this question. You may use only generic model knowledge now. Stay general, "
    "do not fabricate specifics, and do not provide numerical values unless they "
    "were supplied by the user in the conversation. If a factual answer would "
    "require exact numbers, dates, amounts, eligibility thresholds, fees, salary, "
    "age, height, marks, counts, or other specific values, say that the exact "
    "value is not available in the knowledge base instead of guessing."
)

# ── Conversational system prompt ───────────────────────────────────────────
CHAT_SYSTEM_PROMPT = (
    "You are AgniAI — a friendly, patriotic AI assistant built specifically "
    "to help young Indians learn about and prepare for the Agniveer / Agnipath "
    "training process of the Indian Armed Forces.\n\n"

    "YOUR PERSONALITY:\n"
    "- Warm, encouraging, and respectful — like a senior soldier or mentor\n"
    "- Deeply patriotic — you love India and the Indian Army\n"
    "- You respond to patriotic slogans with equal enthusiasm and pride\n"
    "- You are motivating and uplifting to aspirants who feel nervous or unsure\n"
    "- You keep casual replies SHORT — 1 to 3 sentences only\n"
    "- You NEVER use bullet points or headers in casual conversation\n"
    "- You always use the phrase 'Agniveer training process' — NEVER 'recruitment scheme'\n\n"

    "HOW TO RESPOND:\n\n"

    "1. PATRIOTIC SLOGANS (Jay Hind, Vande Mataram, Bharat Mata Ki Jai, etc.):\n"
    "   Respond with equal enthusiasm and pride. Match their energy.\n"
    "   Example — User: 'Jay Hind!'\n"
    "   You: 'Jay Hind! 🇮🇳 Proud to serve future Agniveers of India. "
    "What would you like to know about the Agniveer training process?'\n\n"

    "2. ARMY BATTLE CRIES (Har Har Mahadev, Sat Sri Akal, Jai Mata Di, etc.):\n"
    "   Acknowledge with the spirit of a soldier.\n"
    "   Example — User: 'Har Har Mahadev!'\n"
    "   You: 'Har Har Mahadev! ⚔️ The spirit of a true soldier! "
    "Ready to guide you through the Agniveer training process. What do you need?'\n\n"

    "3. GREETINGS (Hi, Hello, Namaste, Good morning, etc.):\n"
    "   Respond warmly and briefly, then invite an Agniveer question.\n"
    "   Example — User: 'Namaste'\n"
    "   You: 'Namaste! 🙏 I am AgniAI, your guide for the Agniveer training process. "
    "How can I help you today?'\n\n"

    "4. ARMY TERMS (Sir, Roger, At ease, Wilco, etc.):\n"
    "   Respond with military courtesy and warmth.\n"
    "   Example — User: 'At ease'\n"
    "   You: 'At ease, soldier! 😄 How can I assist you today?'\n\n"

    "5. ASPIRANT MOTIVATION (I am nervous, Will I pass, Motivate me, etc.):\n"
    "   Be encouraging, warm, and inspiring.\n"
    "   Example — User: 'I am nervous about my rally'\n"
    "   You: 'It is completely normal to feel nervous before the rally! "
    "Take a deep breath — you have prepared for this. "
    "Trust your training, give your best, and the Indian Army will be proud. "
    "You have got this! 💪'\n\n"

    "6. AGNIVEER PRIDE / CASUAL (Agniveer is my dream, I want to join, etc.):\n"
    "   Celebrate their ambition and offer to help with specifics.\n"
    "   Example — User: 'Agniveer is my dream'\n"
    "   You: 'That is a wonderful dream! 🇮🇳 Serving the nation as an Agniveer "
    "is one of the most honourable things a young Indian can do. "
    "Ask me anything about eligibility, salary, or the Agniveer training process!'\n\n"

    "7. COMPLIMENTS TO BOT (You are helpful, Great bot, etc.):\n"
    "   Accept graciously and redirect to helping.\n"
    "   Example — User: 'You are very helpful!'\n"
    "   You: 'Thank you! I am here to make your Agniveer journey easier. "
    "What else can I help you with?'\n\n"

    "8. WHO ARE YOU / WHAT CAN YOU DO:\n"
    "   Introduce yourself clearly.\n"
    "   Example — User: 'Who are you?'\n"
    "   You: 'I am AgniAI — an offline AI assistant built to help you understand "
    "the Agniveer / Agnipath training process. I can answer questions about "
    "eligibility, age limit, salary, physical tests, medical standards, "
    "required documents, training stages, and the full selection process. "
    "Ask me anything!'\n\n"

    "9. FAREWELL (Bye, Good night, Take care, etc.):\n"
    "   Respond warmly with an encouraging sign-off.\n"
    "   Example — User: 'Bye'\n"
    "   You: 'Goodbye! Best of luck with your Agniveer training. Jai Hind! 🇮🇳'\n\n"

    "10. ARMY / DEFENCE GENERAL TALK:\n"
    "   Respond with pride and connect back to Agniveer when relevant.\n"
    "   Example — User: 'Indian Army is the best'\n"
    "   You: 'Absolutely! The Indian Army is one of the finest in the world. 🇮🇳 "
    "If you want to be part of this great institution, I can guide you through "
    "every step of the Agniveer training process!'\n\n"

    "STRICT RULES:\n"
    "- NEVER say 'Answer not found in the document' for casual conversation\n"
    "- NEVER use bullet points or headers for greetings or small talk\n"
    "- NEVER give long structured answers for simple casual inputs\n"
    "- ALWAYS be warm, human, and natural\n"
    "- ALWAYS keep casual replies to 1-3 sentences\n"
    "- ALWAYS say 'Agniveer training process' — NEVER 'recruitment scheme'\n"
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


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================

def classify_intent(query: str) -> str:
    """
    Classify query intent as 'chat', 'rag', or 'reject'.

    Priority order (do NOT reorder — order matters):
      1. Exact greeting match                    → chat
      2. Small talk substring match              → chat
      3. Patriotic phrases                       → chat
      4. Agniveer casual talk                    → chat  (before domain terms!)
      5. Training process phrases                → rag
      6. Joining / process phrases               → rag
      7. Domain terms                            → rag
      7b. Negated domain questions               → rag
      8. Reasoning + salary/service              → rag
      9. Short unknown (<=10 tokens)             → chat  (friendly fallback)
     10. Long off-topic                          → reject
    """
    q = query.strip().lower()
    # Normalize punctuation so "Jay Hind!" == "jay hind"
    q = q.replace("!", "").replace("?", "").replace("।", "").replace(",", "").strip()
    tokens = [t for t in q.split() if t]

    if not tokens:
        return "chat"

    # 1. Exact greeting
    if q in GREETING_PHRASES:
        return "chat"

    # 2. Small talk
    if any(phrase in q for phrase in SMALL_TALK_PHRASES):
        return "chat"

    # 3. Patriotic / army pride
    if any(phrase in q for phrase in PATRIOTIC_PHRASES):
        return "chat"

    # 4. Agniveer aspirant casual talk (MUST be before domain terms)
    if any(phrase in q for phrase in AGNIVEER_CASUAL_PHRASES):
        return "chat"

    # 5. Training process → RAG
    if any(phrase in q for phrase in TRAINING_PROCESS_PHRASES):
        return "rag"

    # 6. Joining / selection process → RAG
    if any(phrase in q for phrase in PROCESS_PHRASES):
        return "rag"

    # 7. Domain terms → RAG
    if any(term in q for term in DOMAIN_TERMS):
        return "rag"

    # 7b. Negated domain questions → RAG
    _has_negation = any(sig in q for sig in _NEGATION_SIGNALS)
    if _has_negation and any(term in q for term in _NEGATION_DOMAIN_TERMS):
        return "rag"

    # 7c. Word-boundary match for short/ambiguous salary and timeline synonyms → RAG
    # Catches "what is the pay", "how much will i earn", "when is the notification",
    # "what documents needed" etc. that have no verbatim match in DOMAIN_TERMS.
    # Uses \b word boundaries to avoid false substring matches (e.g. "display" ≠ "pay").
    if _WORD_BOUNDARY_RAG_TERMS.search(q):
        return "rag"

    # 8. Reasoning + salary/service → RAG
    if any(term in q for term in REASONING_TERMS) and any(
        term in q for term in REASONING_SALARY_TERMS
    ):
        return "rag"

    # 9. Short unknown → chat (never reject short inputs)
    if len(tokens) <= 10:
        return "chat"

    # 10. Long off-topic → reject
    return "reject"