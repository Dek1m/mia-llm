-- LLM DDL: индексы для агентов
CREATE INDEX IF NOT EXISTS idx_llm_agents_type ON llm.llm_agents (agent_type);
CREATE INDEX IF NOT EXISTS idx_llm_agents_workspace ON llm.llm_agents (workspace_id);
CREATE INDEX IF NOT EXISTS idx_llm_agents_owner ON llm.llm_agents (owner_id);
