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

-- Пайплайны — общесистемный справочник (belle).
-- Runs — пользовательские данные (belle_workspace_*), создаются в _open_user_repo.
