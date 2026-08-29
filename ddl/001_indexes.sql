-- LLM DDL: индексы для агентов
CREATE INDEX IF NOT EXISTS idx_llm_agents_type ON llm.llm_agents (agent_type);
CREATE INDEX IF NOT EXISTS idx_llm_agents_workspace ON llm.llm_agents (workspace_id);
CREATE INDEX IF NOT EXISTS idx_llm_agents_owner ON llm.llm_agents (owner_id);
CREATE INDEX IF NOT EXISTS idx_llm_providers_kind ON llm.llm_providers (kind);
CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_models_provider_model
    ON llm.llm_models (provider_id, model_id);
CREATE INDEX IF NOT EXISTS idx_llm_models_provider ON llm.llm_models (provider_id);
