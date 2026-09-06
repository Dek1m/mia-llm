-- Окно контекста и режимы reasoning из расширенных ответов OpenAI-совместимых
-- провайдеров (OpenRouter context_length, vLLM max_model_len, LM Studio
-- max_context_length). Плюс персональный режим reasoning у агента.
ALTER TABLE llm.llm_models
    ADD COLUMN IF NOT EXISTS context_length INTEGER;
ALTER TABLE llm.llm_models
    ADD COLUMN IF NOT EXISTS reasoning_modes TEXT;
ALTER TABLE llm.llm_agents
    ADD COLUMN IF NOT EXISTS reasoning_effort TEXT;
