# Systematic Compliance Agent

A multi-agent pipeline that takes a raw consumer financial complaint and hands back a regulatory-cited resolution draft, a Jira ticket, a fairness audit, and a complete decision trace — all without a human touching it unless the system genuinely isn't sure.

Built on LangGraph, Anthropic Claude, and a fine-tuned DistilBERT classifier. Data source: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

---


## Architecture Diagram

![Architecture Diagram](images/architecture.png)

---

## What it does

A complaint comes in over the API. Eight agents handle it in sequence:

1. **Intake** — strips PII using spaCy NER followed by regex (spaCy runs first, on the original text, to avoid re-tagging placeholder tokens like `[SSN]`). Hashes the narrative for dedup.

2. **Classifier** — DistilBERT predicts product category and issue type; Claude assesses severity and compliance risk level. If Claude's risk confidence falls below 0.72, the complaint goes straight to the human review queue rather than continuing. SHAP attributions are computed for the audit trail.

3. **Root Cause** — embeds the complaint and finds the 10 nearest complaints in ChromaDB filtered by product. Runs a Z-score check against a rolling 30-day volume history to flag anomalies. Claude then generates a structured causal chain from the cluster.

4. **Routing** — deterministic rules cover the common cases (e.g. credit card + billing → Credit Operations, Reg Z, 60-day SLA). Anything that doesn't match falls back to Claude. IMMINENT risk overrides all of this and escalates directly to Executive Escalation with a 1-day SLA.

5. **Resolution** — RAG query against the regulatory knowledge base (Reg E, Reg Z, FCRA, ECOA, CFPB Circular 2022-06), then Claude drafts immediate actions, a customer response letter, and preventive recommendations. If this is a revision pass, the failed review's instructions are appended to the system prompt.

6. **Regulatory Review** — Claude scores the resolution on five dimensions (factual accuracy, statutory citations, tone, completeness, timeliness representation) for a 0–100 total. Anything under 80 gets sent back to Resolution with specific failure flags. Maximum two revision attempts before it passes regardless.

7. **Audit** — aggregates regulatory citations from every agent, deduped. Computes fairness metrics against FFIEC ZIP-code demographic proxies: demographic parity difference, equalized odds (TPR/FPR), disparate impact ratio (80% rule), and Gini coefficient. Persists everything to PostgreSQL.

8. **Orchestrator** — LangGraph `StateGraph` wiring all of the above with conditional edges, failure recovery, and human escalation.

---

## Pipeline flow

```
Complaint ──► Intake ──► Classifier ──┬──► Human Review ──► Audit
                                      │
                                      └──► Root Cause ──► Routing ──► Resolution ──► Regulatory Review ──┬──► Audit
                                                                              ▲                          │
                                                                              └──── score < 80 (max 2x) ─┘
```

---

## Compliance risk levels

| Level | Trigger | What happens |
|-------|---------|--------------|
| NONE | No apparent regulatory touch | Standard resolution |
| ADVISORY | Touches a regulated product/process | Document thoroughly |
| MODERATE | Potential policy violation | Escalate within 48 hours |
| ELEVATED | Apparent regulatory violation | Compliance team, 24 hours |
| IMMINENT | Active violation with consumer harm | Overrides routing to Executive Escalation, 1-day SLA |

---

## Tech stack

| Layer | What's used |
|-------|-------------|
| Agent orchestration | LangGraph `StateGraph` |
| LLM | Claude (primary), GPT-4o (fallback) |
| Classification | DistilBERT fine-tuned on CFPB data + SHAP |
| Vector store | ChromaDB (HNSW index) |
| Embeddings | OpenAI `text-embedding-3-small` |
| API | FastAPI + Pydantic v2 |
| Task queue | Celery + Redis |
| Stream ingestion | Apache Kafka |
| Databases | PostgreSQL + Redis |
| Observability | LangSmith, Prometheus, Grafana |
| Experiment tracking | MLflow |
| PII masking | spaCy `en_core_web_lg` + regex |
| Containerization | Docker Compose |
| Testing | pytest (unit + integration), Locust (load) |

---

## Quick start

**Prerequisites:** Docker + Compose, Python 3.11+, Anthropic API key, OpenAI API key (embeddings)

```bash
# 1. Setup
make setup
cp .env.example .env
# add your API keys to .env

# 2. Start infrastructure (Postgres, Redis, Kafka, ChromaDB, MLflow, Prometheus, Grafana)
make up

# 3. Load data
# Download complaints.csv from consumerfinance.gov → data/cfpb_raw/complaints.csv
make ingest    # PII mask + normalize, 50K sample by default
make embed     # Build ChromaDB vector stores
make golden    # Generate 500-complaint golden evaluation set

# 4. Run the API
make api-dev   # http://localhost:8000/docs
```

**Send a complaint:**

```bash
curl -X POST http://localhost:8000/api/v1/complaints/process \
  -H "Content-Type: application/json" \
  -d '{
    "narrative": "I have an unauthorized charge on my credit card disputed for 75 days with no resolution. The bank is still charging interest on the disputed amount and reported it as delinquent to credit bureaus.",
    "state": "CA",
    "zip_code": "90001",
    "submitted_via": "Web"
  }'
```

**Tests:**

```bash
make test-unit         # no external dependencies
make test-integration  # requires API keys + running services
make test-load         # Locust, 1,000+ complaints/hr benchmark
```

---

## Dashboards

| | URL |
|--|-----|
| API | http://localhost:8000/docs |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| MLflow | http://localhost:5000 |

---

## Project layout

```
src/
├── agents/          # orchestrator + 7 agents
├── pipeline/        # data ingestion, PII masking, embeddings, Kafka consumer
├── rag/             # regulatory knowledge base (ChromaDB)
├── classifier/      # DistilBERT training + SHAP inference
├── fairness/        # FairnessMonitor: Gini, disparate impact, equalized odds
├── api/             # FastAPI routes
├── tasks/           # Celery workers
├── observability/   # Prometheus metrics
└── config/          # Pydantic settings

prompts/             # version-controlled prompt templates
scripts/             # data pipeline CLIs
tests/               # unit, integration, load
monitoring/          # Prometheus config, Grafana dashboards
docker-compose.yml
```

---

## Regulatory coverage

The RAG knowledge base includes:

- **CFPB Supervision and Examination Manual** — complaint management program requirements
- **Regulation Z** (12 CFR Part 1026) — credit card billing dispute timelines (60–90 days)
- **Regulation E** (12 CFR Part 1005) — EFT dispute obligations (10/45 days)
- **FCRA** (15 U.S.C. 1681) — credit dispute investigation requirements (30/45 days)
- **ECOA / Regulation B** — non-discrimination, adverse action notice obligations
- **CFPB Circular 2022-06** — UDAAP enforcement guidance

---

## Fine-tuning the classifier

```bash
python -c "
from src.classifier.train import train
train(
    data_path='data/cfpb_processed.parquet',
    output_dir='models/checkpoints/distilbert_cfpb',
    num_epochs=5,
    batch_size=32,
)
"
# tracked in MLflow at http://localhost:5000
```

---

## A few design decisions worth knowing

**PII masking order matters.** spaCy runs on the original text first. If regex ran first and replaced `John Smith` with `[PERSON]`, spaCy would then tag `[PERSON]` as an entity and produce a double-masked mess. The current order avoids that.

**Human escalation is based on Claude's confidence, not DistilBERT's.** DistilBERT product confidence is unreliable without fine-tuning and varies a lot on out-of-distribution text. The 0.72 threshold only fires when Claude's own risk assessment is uncertain — meaning the escalation is anchored to the compliance judgment, not the label prediction.

**The regulatory review loop has a hard cap.** Resolutions that fail twice go to audit regardless of score. Without the cap, a complaint with genuinely ambiguous regulatory fit would loop indefinitely. Two revision attempts is enough to catch formatting and citation issues; if it still fails after that, human review is the right answer.

**IMMINENT risk overrides everything.** When the classifier returns `ComplianceRiskLevel.IMMINENT`, the routing agent ignores all deterministic rules and LLM output and directly assigns `EXECUTIVE_ESCALATION` with a 1-day SLA. This is a hard-coded override, not a routing rule.

---

## Targets

| Metric | Baseline (manual process) | Target |
|--------|--------------------------|--------|
| Classification accuracy | ~58% | > 90% |
| Time to route | 3–5 days | < 15 minutes |
| Regulatory response score | undefined | > 85/100 |
| Root cause identification | ~20% | > 75% |
| Fairness disparity (Gini) | unmeasured | < 0.05 |
| Human escalation rate | 100% | < 12% |
| Throughput | ~20/hr | > 1,000/hr |
