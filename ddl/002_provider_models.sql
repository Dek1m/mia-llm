ALTER TABLE llm.llm_providers ADD COLUMN IF NOT EXISTS description TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_models_provider_model
    ON llm.llm_models (provider_id, model_id);
CREATE INDEX IF NOT EXISTS idx_llm_models_provider ON llm.llm_models (provider_id);
