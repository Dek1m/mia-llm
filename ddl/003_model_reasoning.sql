ALTER TABLE llm.llm_models
    ADD COLUMN IF NOT EXISTS supports_reasoning BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE llm.llm_models
    ADD COLUMN IF NOT EXISTS reasoning_enabled BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE llm.llm_models
    ADD COLUMN IF NOT EXISTS reasoning_effort TEXT;
