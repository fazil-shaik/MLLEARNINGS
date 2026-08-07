-- OpsPilot schema — run once against your NeonDB database.
-- LangGraph's Postgres checkpointer creates its own tables automatically
-- (checkpoints, checkpoint_writes, etc.) on first run — no need to define them here.

CREATE TABLE IF NOT EXISTS leads (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    company         TEXT,
    email           TEXT,
    source          TEXT,
    score           INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'new',        -- new | enriched | contacted | qualified | rejected
    enriched_data   JSONB DEFAULT '{}',
    outreach_draft  TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tickets (
    id              SERIAL PRIMARY KEY,
    customer        TEXT NOT NULL,
    subject         TEXT,
    description     TEXT,
    category        TEXT,
    priority        TEXT DEFAULT 'normal',      -- low | normal | high | urgent
    status          TEXT DEFAULT 'open',        -- open | resolved | escalated
    resolution      JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ops_tasks (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT,
    steps           JSONB DEFAULT '[]',
    status          TEXT DEFAULT 'pending',     -- pending | in_progress | done | failed
    result          JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Full audit trail of every agent execution: which model, what it cost, how long.
CREATE TABLE IF NOT EXISTS agent_runs (
    id              SERIAL PRIMARY KEY,
    thread_id       TEXT,                       -- LangGraph thread/session id
    crew_name       TEXT,                        -- sales | support | ops
    agent_name      TEXT,
    model_used      TEXT,
    input_summary   TEXT,
    output_summary  TEXT,
    tokens_used     INTEGER,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    role            TEXT,                        -- user | agent | system
    content         TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_thread ON agent_runs (thread_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations (session_id);
