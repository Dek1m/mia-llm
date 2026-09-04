"""LLM DB Schema — Schema-first dict для register_schema.

Таблица llm_agents: агенты (системные, пользовательские, workspace).
"""
from __future__ import annotations

from typing import Any

__all__ = ["DB_SCHEMA"]

DB_SCHEMA: dict[str, dict[str, Any]] = {
    "schema": "llm",
    "llm_agents": {
        "columns": {
            "name": "TEXT NOT NULL UNIQUE",
            "agent_type": "TEXT NOT NULL",
            "description": "TEXT",
            "system_prompt": "TEXT",
            "model": "TEXT",
            "settings": "JSONB DEFAULT '{}'::jsonb",
            # UUID без FK: workspace.workspaces живёт в per-user БД, не в belle.
            "workspace_id": "UUID",
            "owner_id": "UUID",
            "is_active": "BOOLEAN DEFAULT TRUE",
            "is_visible": "BOOLEAN NOT NULL DEFAULT TRUE",
            "is_default": "BOOLEAN NOT NULL DEFAULT FALSE",
            "created_at": "TIMESTAMPTZ DEFAULT NOW()",
            "updated_at": "TIMESTAMPTZ DEFAULT NOW()",
        },
    },
    "llm_agent_avatars": {
        "auto_id": False,
        "columns": {
            "agent_id": "UUID PRIMARY KEY REFERENCES llm.llm_agents(id) ON DELETE CASCADE",
            "bytes": "BYTEA NOT NULL",
            "content_type": "VARCHAR(64) NOT NULL",
            "updated_at": "TIMESTAMPTZ DEFAULT NOW()",
        },
    },
    "llm_providers": {
        "columns": {
            "name": "TEXT NOT NULL UNIQUE",
            "kind": "TEXT NOT NULL",
            "vendor": "TEXT NOT NULL",
            "description": "TEXT",
            "base_url": "TEXT",
            "default_model": "TEXT",
            "api_key": "TEXT",
            "oauth_status": "TEXT",
            "is_active": "BOOLEAN DEFAULT TRUE",
            "created_at": "TIMESTAMPTZ DEFAULT NOW()",
            "updated_at": "TIMESTAMPTZ DEFAULT NOW()",
        },
    },
    "llm_models": {
        "columns": {
            "provider_id": "UUID NOT NULL REFERENCES llm.llm_providers(id) ON DELETE CASCADE",
            "model_id": "TEXT NOT NULL",
            "display_name": "TEXT NOT NULL",
            "enabled": "BOOLEAN NOT NULL DEFAULT TRUE",
            "is_available": "BOOLEAN NOT NULL DEFAULT TRUE",
            "supports_reasoning": "BOOLEAN NOT NULL DEFAULT FALSE",
            "reasoning_enabled": "BOOLEAN NOT NULL DEFAULT FALSE",
            "reasoning_effort": "TEXT",
            "created_at": "TIMESTAMPTZ DEFAULT NOW()",
            "updated_at": "TIMESTAMPTZ DEFAULT NOW()",
        },
    },
}
