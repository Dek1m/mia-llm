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
            "workspace_id": "UUID REFERENCES workspace.workspaces(id) ON DELETE CASCADE",
            "owner_id": "UUID REFERENCES auth.users(id)",
            "is_active": "BOOLEAN DEFAULT TRUE",
            "created_at": "TIMESTAMPTZ DEFAULT NOW()",
            "updated_at": "TIMESTAMPTZ DEFAULT NOW()",
        },
    },
}
