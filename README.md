# Systematic Compliance Agent
## Multi-Agent Fintech AI Pipeline

A production-grade, multi-agent AI system that autonomously classifies, routes, investigates, and resolves consumer financial complaints at scale — with full explainability for regulatory audit trails.

---

## Architecture Overview

```
Consumer Complaint
       │
       ▼
┌─────────────┐
│ Intake Agent│  PII masking (spaCy NER + regex), deduplication (SHA-256), normalization
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Classifier Agent │  DistilBERT (product/issue) + Claude LLM (severity/compliance risk)
│                  │  ChromaDB few-shot retrieval · SHAP feature attribution
└──────┬───────────┘
       │
       ├── confidence < 0.72 ──► Human Review Queue
       │
       ▼
┌────────────────────┐
│ Root Cause Agent   │  HDBSCAN semantic clustering · Z-score anomaly detection
│                    │  LLM causal chain generation (structured JSON)
└──────┬─────────────┘
       │
       ▼
┌──────────────┐
│ Routing Agent│  Rule-based fast path + LLM fallback · Priority scoring
│              │  Mock Jira/ServiceNow ticket creation · SLA deadline assignment
└──────┬───────┘
       │
       ▼
┌────────────────────┐
│ Resolution Agent   │  ReAct + RAG over regulatory corpus (Reg E/Z, FCRA, ECOA)
│                    │  Customer response draft · Preventive recommendations
└──────┬─────────────┘
       │
       ▼
┌────────────────────────┐
│ Regulatory Review Agent│  LLM-as-judge (5-dimension rubric · 0-100 score)
│                        │  Score < 80 → return to Resolution Agent (max 2 revisions)
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Audit & Explainability │  Full decision trace · SHAP attributions · Regulatory citations
│ Agent                  │  Fairness metrics (Gini, disparate impact, equalized odds)
│                        │  PostgreSQL persistence · LangSmith tracing
└────────────────────────┘
```

**Orchestration**: LangGraph StateGraph with conditional edges, interrupt handlers, and automatic human escalation

---

## Success Criteria

| Metric | Baseline | Target |
|--------|----------|--------|
| Classification accuracy | ~58% | **> 90%** |
| Time to route | 3-5 days | **< 15 minutes** |
| Regulatory response score | Undefined | **> 85/100** |
| Root cause identification | ~20% | **> 75%** |
| Fairness disparity (Gini) | Unmeasured | **< 0.05** |
| Human escalation rate | 100% | **< 12%** |
| Throughput | ~20/hr | **> 1,000/hr** |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph StateGraph |
| LLM | Anthropic Claude (primary) · GPT-4o (fallback) |
| ML Classification | HuggingFace DistilBERT (fine-tuned) |
| Vector Store | ChromaDB (persistent, HNSW index) |
| Embeddings | OpenAI text-embedding-3-small |
| API | FastAPI + Pydantic v2 + Uvicorn |
| Task Queue | Celery + Redis |
| Stream Ingestion | Apache Kafka |
| Databases | PostgreSQL + Redis |
| Observability | LangSmith + Prometheus + Grafana |
| Experiment Tracking | MLflow |
| Fairness | SHAP + custom Gini module |
| PII Masking | spaCy en_core_web_lg + regex |
| Containerization | Docker Compose |
| Testing | pytest + Locust |

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- Anthropic API key
- OpenAI API key (for embeddings)

### 1. Environment Setup

```bash
make setup
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Infrastructure

```bash
make up
# Services: Postgres, Redis, Kafka, ChromaDB, MLflow, Prometheus, Grafana
```

### 3. Initialize Data Pipeline

```bash
# Download CFPB data from consumerfinance.gov → data/cfpb_raw/complaints.csv
make ingest          # Ingest + PII mask + normalize (50K sample by default)
make embed           # Build ChromaDB vector stores
make golden          # Generate 500-complaint golden evaluation set
```

### 4. Run API

```bash
make api-dev         # Development server with hot reload
# Swagger UI: http://localhost:8000/docs
```

### 5. Submit a Complaint (Demo)

```bash
curl -X POST http://localhost:8000/api/v1/complaints/process \
  -H "Content-Type: application/json" \
  -d '{
    "narrative": "I have an unauthorized charge on my credit card that has been disputed for 75 days with no resolution. The bank is still charging interest on the disputed amount and has reported it as delinquent to credit bureaus.",
    "state": "CA",
    "zip_code": "90001",
    "submitted_via": "Web"
  }'
```

### 6. Run Tests

```bash
make test-unit         # Unit tests (no external dependencies)
make test-integration  # Integration tests (requires API keys + services)
make test-load         # Locust load test (1,000+ complaints/hr benchmark)
```

---

## Dashboards

| Dashboard | URL |
|-----------|-----|
| API Swagger UI | http://localhost:8000/docs |
| Grafana Pipeline Health | http://localhost:3000 (admin/admin) |
| Prometheus Metrics | http://localhost:9090 |
| MLflow Experiments | http://localhost:5000 |

---

## Project Structure

```
├── src/
│   ├── agents/          # 8 specialized agents (intake, classifier, root_cause, routing,
│   │                    #   resolution, regulatory_review, audit, orchestrator)
│   ├── pipeline/        # Data ingestion, PII masking, embeddings, Kafka consumer
│   ├── rag/             # Regulatory knowledge base (ChromaDB-backed)
│   ├── classifier/      # DistilBERT fine-tuning + SHAP inference
│   ├── fairness/        # Gini, disparate impact, equalized odds monitoring
│   ├── api/             # FastAPI routes (complaints, audit)
│   ├── tasks/           # Celery async task queue
│   ├── observability/   # Prometheus metrics
│   └── config/          # Pydantic settings
├── prompts/             # Version-controlled LLM prompt templates
├── scripts/             # Data pipeline CLI scripts
├── tests/               # Unit, integration, load tests
├── monitoring/          # Prometheus config + Grafana dashboards
└── docker-compose.yml   # Full local deployment
```

---

## Regulatory Coverage

The RAG knowledge base covers:
- **CFPB Supervision and Examination Manual** — Complaint management program requirements
- **Regulation Z** (12 CFR Part 1026) — Credit card billing dispute timelines (60-90 days)
- **Regulation E** (12 CFR Part 1005) — EFT dispute obligations (10/45 days)
- **FCRA** (15 U.S.C. 1681) — Credit dispute investigation requirements (30/45 days)
- **ECOA / Regulation B** — Non-discrimination, adverse action notice obligations
- **CFPB Circular 2022-06** — UDAAP enforcement guidance

---

## Compliance Risk Levels

| Level | Description | Action |
|-------|-------------|--------|
| NONE | No regulatory implications | Standard resolution |
| ADVISORY | Touches regulated activity, no apparent violation | Document thoroughly |
| MODERATE | Potential policy violation | Escalate within 48 hours |
| ELEVATED | Apparent regulatory violation | Escalate to Compliance in 24 hours |
| IMMINENT | Active violation with consumer harm | Immediate escalation |

---

## Fairness Monitoring

The Audit Agent computes fairness metrics using FFIEC ZIP-code demographic proxies:

- **Demographic Parity Difference** (threshold: < 0.05)
- **Equalized Odds** TPR/FPR parity (max diff: < 0.05)
- **Disparate Impact Ratio** using the 80% rule (threshold: > 0.80)
- **Gini Coefficient** of resolution outcomes (threshold: < 0.05)

---

## Fine-Tuning the Classifier

```bash
# After ingestion and before evaluation:
python -c "
from src.classifier.train import train
train(
    data_path='data/cfpb_processed.parquet',
    output_dir='models/checkpoints/distilbert_cfpb',
    num_epochs=5,
    batch_size=32,
)
"
# Results logged to MLflow at http://localhost:5000
```

---

## Operating Principles

- **No raw PII in logs**: All narrative data masked before any LLM or storage operation
- **Prompt versioning**: Every prompt template in `prompts/` is treated as code (git-tracked)
- **Human escalation is a feature**: The system surfaces its own uncertainty
- **Fallback at every layer**: Each agent has defined fallback behavior for failures
- **Audit by default**: Every pipeline run produces a regulator-ready audit trail
- **Cost management**: ~0.3-0.7M input tokens per 10K complaints; token budgets enforced per complaint

---

## Week-by-Week Roadmap

| Week | Focus |
|------|-------|
| 1 | Foundation: Docker, Kafka, PII pipeline, DistilBERT v1, LangGraph scaffold |
| 2 | Core agents: Root cause, routing, resolution, regulatory review |
| 3 | Scale: Celery workers, HNSW tuning, SHAP, fairness module, Grafana |
| 4 | Evaluation: Golden set, ablation study, regulator audit trails, demo |

---

*Systematic Compliance Agent | R&D Competition | Senior AI Engineer, Financial Services*
