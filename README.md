---
language: en
tags:
  - rag
  - ollama
  - offline
  - agniveer
  - agnipath
  - chatbot
  - text2sql
  - retrieval-augmented-generation
  - faiss
  - sentence-transformers
  - admin-intelligence
  - prometheus
license: mit
---

# AgniAI — Offline Agniveer RAG Chatbot + Admin Intelligence Engine

AgniAI is an enterprise-grade, fully offline platform featuring **two powerful AI engines in a unified service**:

1. **User RAG Chatbot** — Interactive, domain-grounded retrieval system for **Agniveer & Agnipath recruitment, eligibility, salaries, and training** guidelines. Powered by hybrid dense/sparse vector search (FAISS + BM25) and local Ollama LLMs.
2. **Admin Intelligence Engine** — Natural language SQL command center for military & administrative officers. Features high-precision intent classification, entity resolution, Text2SQL query planning, AST SQL generation, DB execution via SQL Server (`pyodbc`), and side-by-side multi-way comparative analytics.

Both engines run 100% locally with zero cloud dependencies or external API key requirements.

---

## Key Highlights & Recent Capabilities

- 📊 **SQL Analytics & Text2SQL Engine**: Direct, secure SQL generation and execution across 12+ administrative domains (`Performance`, `Attendance`, `Leave`, `Medical`, `Verification`, `Equipment`, `Distribution`, `Skills`, `Schedule`, `OrgHierarchy`, `Disqualified`, `PersonalDetails`).
- 🏢 **Dynamic Unit & Entity Resolution**: Spoken company names (`Lak`, `Jas`, `Arora`, `Krishna`, `Mahadev`) and platoon designations (`Platoon 1`, `Platoon 2`) dynamically match database identities via `resolve_company_id_from_name` and `resolve_platoon_id_from_name`.
- 🔍 **Police Verification Intelligence**:
  - **Sent**: `pv.Status = 'Sent'`
  - **Not Responded**: `pv.Status = 'Sent' AND pv.ReceivedDate IS NULL`
  - **Verified**: `pv.Status IN ('Verified', 'Completed')`
  - **Pending**: `pv.Status = 'Rejected' OR pv.AgniveerId IS NULL`
- 🚨 **Absconded Agniveer Tracking**: Full extraction of open-ended absconded leave records (`l.IsAbscondedLeave = 1`) without requiring closed `ToDate` or `IsActive = 1` flags.
- ⚡ **Threshold Agniveers Pipeline**: CTE pipeline with `ROW_NUMBER() OVER (PARTITION BY AgniveerNo ORDER BY CASE WHEN Reason = 'Continuous 40-44 days' THEN 1 WHEN Reason = 'Total 55-59 days' THEN 2 ELSE 3 END)` window deduplication for continuous (40-44 days) vs. cumulative total (55-59 days) thresholds.
- 🎯 **Summary vs. Detailed Modes**:
  - **Summary Mode** (Default): Returns concise aggregate metrics (e.g. `SUM(MarksObtained) AS BestTotal`).
  - **Detailed Mode** (Triggered by `"detailed"`, `"in detail"`, `"subsection"`, `"full breakdown"`): Unrolls section and subsection-wise breakdowns without row-level mid-Agniveer truncation.
- 🔀 **Side-by-Side Multi-Way Comparison**: Performs N-way side-by-side comparative analysis (e.g., `"Compare top performers in Lak company and Jas company"`) with per-operation entity scope isolation and distinct side labeling.

---

## Architecture Overview

```
                                 ┌──────────────────────────────┐
     User RAG Question ─────────▶│  /api/chat   (RAG Pipeline)  │──▶ FAISS+BM25 ─▶ Ollama ─▶ Grounded Answer
                                 └──────────────────────────────┘

                                 ┌──────────────────────────────┐
     Admin Command Question ────▶│       /api/admin/chat        │
                                 │     (admin_pipeline.py)      │
                                 └──────────────┬───────────────┘
                                                │
                                  plan_query / classify_intent
                                                │
                                 ┌──────────────▼───────────────┐
                                 │   SQL Execution Engine       │
                                 │   - performance_executor.py  │
                                 │   - sql_executor.py          │
                                 │   - pyodbc (SQL Server)      │
                                 └──────────────┬───────────────┘
                                                │
                                  combine_results / report_gen
                                                │
                                 ┌──────────────▼───────────────┐
                                 │       Response Builder       │──▶ Grounded Report + Visual Widgets
                                 └──────────────────────────────┘
```

---

## Core System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.9+ (3.11/3.13 supported) |
| **Database** | MS SQL Server (via `pyodbc`) |
| **Ollama** | 0.1.x+ (Mistral, Llama 3, Phi-3, etc.) |
| **RAM** | 8 GB minimum (16 GB recommended) |
| **Disk** | ~5 GB for local model weights |

---

## Quick Setup Guide

### 1. Install & Boot Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download installer from https://ollama.com/download
ollama pull mistral:7b-instruct-q4_K_M
ollama serve
```

### 2. Install Python Dependencies

```bash
git clone https://github.com/florencygajera/AgniAI.git
cd AgniAI

python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux:   source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and set your local SQL Server connection string:

```env
OLLAMA_MODEL=mistral:7b-instruct-q4_K_M
OLLAMA_BASE_URL=http://127.0.0.1:11434
SQL_READONLY_CONN=Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=AgniAIDb;Trusted_Connection=yes;
```

### 4. Run the API Server

```bash
python app.py
# REST API starts at http://0.0.0.0:5000
```

---

## Admin Intelligence Supported Modules

| Module | Core Capabilities |
|--------|-------------------|
| **Performance** | BPET, PPT, Firing, Drill, Best Attempts, Section & Subsection breakdowns, Top/Bottom N Agniveers. |
| **Attendance** | Daily present/absent headcount, monthly/weekly attendance rates, date range filtering. |
| **Leave** | Absconded leave tracking, 40-44 / 55-59 day threshold alerts, Annual/Sick/Medical leaves. |
| **Medical** | Dynamic BMI formula computation (Normal, Overweight, Obese, Unfit), Blood Group distribution, Hospitalizations. |
| **Verification** | Police verification status tracking (Pending, Sent, Verified, Not Responded). |
| **Equipment** | Issued vs. Returned equipment, currently holding equipment, item conditions & remarks. |
| **Skills & Sports** | Sport rosters (Volleyball, Cricket, Football, Kabaddi), skill distributions by class (Sikh, Dogra, etc.). |
| **Schedule** | Daily training schedules, cold-weather schedule routines, company & platoon training schedules. |
| **Org Hierarchy** | Current & historical company commanders, platoon commanders, commanding officer tenures. |
| **Disqualified** | Disqualified Agniveer rosters, disqualification reasons and dates. |
| **Personal Details** | State of origin, date of joining, class, father name, personal attributes. |

---

## Testing & Quality Assurance

AgniAI includes an extensive pytest suite validating query classification, Text2SQL execution, precision entity resolution, and reliability:

```bash
# Run complete test suite (210+ test cases)
pytest tests/test_query_planner.py tests/test_sql_executor.py tests/test_intent_precision.py -v
```

---

## Privacy & Security

- **100% Offline Computation**: All RAG embeddings, vector searches, and LLM inference execute locally.
- **SQL Guardrails**: Strict read-only execution layer (`run_readonly`) with AST schema guards preventing data mutation.
- **Audit Scrubbing**: Rotating JSON audit logger scrubs prompts, payloads, and secrets.
