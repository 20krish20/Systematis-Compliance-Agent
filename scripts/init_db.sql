-- PostgreSQL initialization: audit records and metadata tables

CREATE TABLE IF NOT EXISTS audit_records (
    id                  SERIAL PRIMARY KEY,
    complaint_id        VARCHAR(64)  UNIQUE NOT NULL,
    pipeline_run_id     VARCHAR(64)  NOT NULL,
    agent_steps         JSONB,
    total_duration_ms   INTEGER,
    regulatory_citations JSONB,
    fairness_metrics    JSONB,
    human_escalated     BOOLEAN      DEFAULT FALSE,
    escalation_reason   TEXT,
    created_at          TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_complaint_id    ON audit_records(complaint_id);
CREATE INDEX IF NOT EXISTS idx_audit_created_at      ON audit_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_human_escalated ON audit_records(human_escalated);

CREATE TABLE IF NOT EXISTS complaint_metrics (
    id              SERIAL PRIMARY KEY,
    window_start    TIMESTAMPTZ NOT NULL,
    window_end      TIMESTAMPTZ NOT NULL,
    total_processed INTEGER     DEFAULT 0,
    total_escalated INTEGER     DEFAULT 0,
    avg_duration_ms FLOAT,
    avg_reg_score   FLOAT,
    escalation_rate FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fairness_snapshots (
    id                      SERIAL PRIMARY KEY,
    snapshot_at             TIMESTAMPTZ DEFAULT NOW(),
    demographic_parity_diff FLOAT,
    disparate_impact_ratio  FLOAT,
    gini_index              FLOAT,
    violation_count         INTEGER     DEFAULT 0,
    group_breakdown         JSONB
);
