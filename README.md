# LLM Module for Mia Framework

Абстракция над LLM-провайдерами + управление определениями агентов.

## Features

- **Provider abstraction** — единый интерфейс для разных бэкендов (OpenAI-compatible, llama.cpp, и т.д.)
- **Agent Definitions** — шаблоны агентов (system prompt, model, tools, parameters)
- **Chat Completions** — синхронные и потоковые вызовы
- **Tool calling** — поддержка function/tool calls
- **Интеграция с Task System** — все вызовы через `@task`
- **Sessions-aware** (опционально) — может принимать session_id и подтягивать историю

## Installation

```bash
git clone https://github.com/Dek1m/mia-llm.git
cd mia-llm
pip install -e .
```

## License

MIT
