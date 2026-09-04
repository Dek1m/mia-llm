CREATE TABLE IF NOT EXISTS llm.pipelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    slug TEXT NOT NULL UNIQUE,
    purpose TEXT,
    caps JSONB NOT NULL DEFAULT '{}'::jsonb,
    rev INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS llm.pipeline_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_id UUID NOT NULL REFERENCES llm.pipelines(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    middleware_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    config JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS pipeline_steps_ord
    ON llm.pipeline_steps (pipeline_id, ord);

CREATE TABLE IF NOT EXISTS llm.runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_id UUID REFERENCES llm.pipelines(id) ON DELETE SET NULL,
    session_id UUID,
    workspace_id UUID,
    agent_id UUID,
    user_id UUID,
    status TEXT NOT NULL,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cache_tokens INTEGER NOT NULL DEFAULT 0,
    cache_hits INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS llm_runs_session_created
    ON llm.runs (session_id, created_at DESC);
