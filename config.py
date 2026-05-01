"""Central configuration for AgniAI."""

from __future__ import annotations

import os
import re
from pathlib import Path

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
MODEL_MAX_CONTEXT_TOKENS = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

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
# Set this in production to protect destructive endpoints (e.g. /api/reset_index)
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
    "elaborate": int(os.getenv("MAX_TOKENS_ELABORATE", "500")),
    "detail":    int(os.getenv("MAX_TOKENS_DETAIL",    "800")),
}
MAX_TOKENS_DEFAULT  = int(os.getenv("MAX_TOKENS_DEFAULT",  "500"))
TOKEN_SAFETY_BUFFER = int(os.getenv("TOKEN_SAFETY_BUFFER", "100"))


# =============================================================================
# INTENT CLASSIFICATION PHRASE LISTS
#
# IMPORTANT — priority order in classify_intent():
#   chat phrases are checked BEFORE domain terms so that casual expressions
#   like "i want to join agniveer" go to chat, not RAG.
# =============================================================================

# ── Greetings — exact match ────────────────────────────────────────────────
GREETING_PHRASES = {
    # Basic
    "hi", "hello", "hey", "hii", "hiii", "heyy", "heya",
    "hola", "howdy", "yo", "sup", "wassup", "whatsup",
    # Farewells
    "bye", "goodbye", "good bye", "see you", "see ya",
    "take care", "good night", "goodnight", "good day",
    # Time-based
    "good morning", "good afternoon", "good evening",
    # Thanks
    "thanks", "thank you", "thank u", "thankyou",
    "ty", "thx", "thanks a lot", "thank you so much",
    "many thanks", "dhanyawad", "shukriya", "meherbani",
    # Acknowledgements
    "ok", "okay", "ok thanks", "okay thanks", "alright",
    "got it", "understood", "sure", "noted", "cool",
    "great", "nice", "awesome", "wonderful", "amazing",
    "fantastic", "excellent", "perfect", "good",
    "well done", "good job", "nice work", "keep it up", "carry on",
    # Patriotic short exact phrases
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
    # Hindi greetings
    "namaste", "namaskar", "pranam", "namasté",
    "jai shree ram", "jai shri ram", "ram ram", "jai siya ram",
    # Army terms
    "yes sir", "sir", "ma'am", "mam",
    "attention", "at ease", "dismissed",
    "roger", "roger that", "copy that", "wilco",
    "over", "out", "fall in", "stand easy",
    # Bot compliments (short exact)
    "you are helpful", "youre helpful", "you're great",
    "you are great", "you are good", "youre good",
    "nice bot", "good bot", "helpful bot", "great bot",
}

# ── Small talk — partial / substring match ─────────────────────────────────
SMALL_TALK_PHRASES = (
    # How are you variants
    "how are you", "how r you", "how are u", "how r u",
    "how do you do", "how is it going", "how's it going",
    "hows it going", "how are things", "how are you doing",
    "are you okay", "you okay", "u okay",
    # What's up
    "what's up", "whats up", "what is up",
    # Identity questions
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
    # Feelings
    "i am happy", "i am sad", "i feel good", "i feel bad",
    "i am bored", "i am tired", "i am stressed", "i am confused",
    "i need help", "help me", "please help", "help",
    # Filler
    "nice to meet you", "nice to meet u",
    "pleased to meet you", "great to meet you",
    "good to meet you", "glad to meet you",
    "you are amazing", "you are wonderful", "you are awesome",
    "i like you", "i love you", "i love this",
    "this is great", "this is helpful",
    "very helpful", "so helpful", "great help", "big help",
    # Availability
    "you there", "are you there", "you available",
    "are you available", "are you online", "u there",
    "hello there", "anyone there", "is anyone there",
    # Motivation asks
    "motivate me", "give me motivation", "i need motivation",
    "inspire me", "give me inspiration", "encourage me", "cheer me up",
    # Farewells / wishes
    "have a good day", "have a nice day", "have a great day",
    "have a wonderful day", "enjoy your day",
    "take care of yourself", "stay safe", "stay healthy",
    "all the best", "best of luck", "good luck",
    "wish me luck", "fingers crossed",
)

# ── Patriotic / army pride — partial match ─────────────────────────────────
PATRIOTIC_PHRASES = (
    # National slogans
    "jay hind", "jai hind", "vande mataram",
    "bharat mata ki jai", "jai bharat",
    "inquilab zindabad", "jai jawan jai kisan",
    "jai jawan", "jai kisan", "jai vigyan",
    "mera bharat mahan", "hindustan zindabad",
    "bharat zindabad", "india zindabad",
    # Army values / heritage words
    "shaurya", "veerta", "parakram", "balidan",
    "shaheed", "sainik", "sena", "fauj", "fauji",
    "desh seva", "rashtra seva", "desh bhakti", "deshbhakti",
    "watan", "tiranga", "tricolor", "national flag",
    # Occasions
    "republic day", "independence day", "army day",
    "vijay diwas", "kargil vijay diwas",
    # Regiment / corps battle cries
    "jai mata di", "durga mata ki jai", "har har mahadev",
    "bum bum bhole", "sat sri akal", "waheguru ji ka khalsa",
    "jai rajputana", "rajputana rifles", "jai mahakal",
    "bol bajrang bali ki jai",
    "indian army zindabad", "army zindabad",
    # Pride / motivational
    "proud to be indian", "proud of indian army",
    "salute to army", "salute to soldiers",
    "respect the army", "army is great",
    "soldiers are heroes", "our army is the best",
    "indian army is best", "love indian army",
    "support our troops", "army rocks", "army is life",
)

# ── Agniveer aspirant casual talk ──────────────────────────────────────────
# Checked BEFORE DOMAIN_TERMS — casual phrases go to chat, not RAG.
AGNIVEER_CASUAL_PHRASES = (
    # Identity / aspiration
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
    # Motivation / emotions
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
    "i failed",
    "i got rejected",
    "i did not pass",
    "i did not qualify",
    "better luck next time",
    "i will try again",
    "never give up",
    # General agniveer talk
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
)

# ── Domain terms — RAG trigger ─────────────────────────────────────────────
# NOTE: "agniveer" and "training" alone are intentionally NOT here because
# casual phrases containing those words (e.g. "i love agniveer") should be
# caught by AGNIVEER_CASUAL_PHRASES first and go to chat.
DOMAIN_TERMS = (
    "age limit",
    "eligibility",
    "salary",
    "pay scale",
    "stipend",
    "in hand",
    "medical",
    "pft",
    "physical fitness test",
    "insurance",
    "ncc certificate",
    "document required",
    "seva nidhi",
    "benefit",
    "package",
    "rally",
    "physical test",
    "fitness test",
    "written exam",
    "cee",
    "common entrance",
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
    "posting",
    "deployment",
    "pass out",
    "passout",
    "allotment",
    "completion certificate",
    "domicile",
    "admit card",
    "merit list",
    "cut off",
    "height requirement",
    "chest measurement",
    "weight requirement",
    "1.6 km run",
    "beam",
    "long jump",
    "matric",
    "class 10",
    "marksheet",
    "aadhaar",
    "character certificate",
    "caste certificate",
    "agnipath",
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
    "how long",
    "how many weeks",
    "how many months",
    "how many days",
    "duration of",
    "length of",
    "when does",
    "when will",
    "what happens after",
    "what happens during",
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

SYSTEM_PROMPT_SHORT     = STRICT_RAG_PROMPT
SYSTEM_PROMPT_ELABORATE = STRICT_RAG_PROMPT
SYSTEM_PROMPT_DETAIL    = STRICT_RAG_PROMPT
SYSTEM_PROMPT           = STRICT_RAG_PROMPT

REFERENCE_FALLBACK = "Answer not found in the document."

# ── Conversational system prompt ───────────────────────────────────────────
# Used for ALL non-RAG (chat) responses.
CHAT_SYSTEM_PROMPT = (
    "You are AgniAI — a friendly, patriotic AI assistant built specifically "
    "to help young Indians learn about and prepare for the Agniveer / Agnipath "
    "training process of the Indian Armed Forces.\n\n"

    "YOUR PERSONALITY:\n"
    "- Warm, encouraging, and respectful — like a senior soldier or mentor\n"
    "- Deeply patriotic — you love India and the Indian Army\n"
    "- Emotionally intelligent — you understand feelings like fear, doubt, excitement, and pride\n"
    "- You respond with empathy first, then guidance when emotions are involved\n"
    "- You motivate and uplift aspirants who feel nervous or unsure\n"
    "- You keep casual replies SHORT — 1 to 3 sentences only\n"
    "- You NEVER use bullet points or headers in casual conversation\n"
    "- You always use the phrase 'Agniveer training process' — NEVER 'recruitment scheme'\n\n"

    "RESPONSE DECISION LOGIC:\n"
    "- If relevant information exists in the knowledge base, you MUST answer using that information accurately.\n"
    "- If the question is factual but not found in the knowledge base, use your own reasoning and general knowledge.\n"
    "- If the user expresses emotion (nervous, fear, confusion, excitement), FIRST acknowledge the feeling, THEN guide.\n"
    "- If the input is casual or conversational, respond naturally like a human mentor.\n"
    "- NEVER say 'answer not found in the document'. Always provide the best helpful response.\n"
    "- Balance correctness for factual answers and emotional connection for conversations.\n\n"

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
    "   Acknowledge feelings first, then encourage and guide.\n"
    "   Example — User: 'I am nervous about my rally'\n"
    "   You: 'I understand you feel nervous — that’s completely normal before a rally. "
    "Trust your preparation, stay focused, and give your best. You’ve got this! 💪'\n\n"

    "6. AGNIVEER PRIDE / CASUAL (Agniveer is my dream, I want to join, etc.):\n"
    "   Celebrate their ambition and offer to help.\n"
    "   Example — User: 'Agniveer is my dream'\n"
    "   You: 'That is a wonderful dream! 🇮🇳 Serving the nation as an Agniveer "
    "is one of the most honourable paths. Ask me anything about the Agniveer training process!'\n\n"

    "7. COMPLIMENTS TO BOT:\n"
    "   Accept politely and continue helping.\n\n"

    "8. WHO ARE YOU:\n"
    "   Introduce clearly and confidently.\n\n"

    "9. FAREWELL:\n"
    "   Warm, motivating sign-off.\n\n"

    "10. ARMY / DEFENCE TALK:\n"
    "   Respond with pride and connect to Agniveer journey.\n\n"

    "STRICT RULES:\n"
    "- NEVER say 'Answer not found in the document'\n"
    "- NEVER break character as AgniAI\n"
    "- NEVER sound robotic or emotionless\n"
    "- NEVER give long structured answers for casual inputs\n"
    "- ALWAYS keep casual replies to 1–3 sentences\n"
    "- ALWAYS be warm, human, and natural\n"
    "- ALWAYS say 'Agniveer training process' — NEVER 'recruitment scheme'\n"
)


def style_structure_instruction(style: str) -> str:
    style_key = (style or "").strip().lower()
    guidance  = STYLE_OUTPUT_GUIDANCE.get(style_key, STYLE_OUTPUT_GUIDANCE["elaborate"])
    return f"{_PARAGRAPH_RULES}\n\n{guidance}"


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
    """Return text trimmed to the last complete sentence."""
    text = (text or "").strip()
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    matches = list(re.finditer(r"[.!?](?:\s|$)", text))
    if not matches:
        return text
    last_end = matches[-1].end()
    trimmed  = text[:last_end].strip()
    if trimmed and len(trimmed) >= len(text) * 0.70:
        return trimmed
    return text


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================

def classify_intent(query: str) -> str:
    """
    Classify query intent as 'chat', 'rag', or 'reject'.

    Priority order (do NOT reorder — order matters):
      1. Exact greeting match          → chat
      2. Small talk substring match    → chat
      3. Patriotic phrases             → chat
      4. Agniveer casual talk          → chat  (before domain terms!)
      5. Training process phrases      → rag
      6. Joining / process phrases     → rag
      7. Domain terms                  → rag
      8. Reasoning + salary            → rag
      9. Short unknown (<=10 tokens)   → chat  (friendly fallback)
     10. Long off-topic                → reject
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

    # 8. Reasoning + salary → RAG
    if any(term in q for term in REASONING_TERMS) and any(
        term in q for term in REASONING_SALARY_TERMS
    ):
        return "rag"

    # 9. Short unknown → chat (never reject short inputs)
    if len(tokens) <= 10:
        return "chat"

    # 10. Long off-topic → reject
    return "reject"