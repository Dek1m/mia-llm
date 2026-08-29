"""LLM Repository — CRUD для таблицы llm_agents."""
from __future__ import annotations

from typing import Any

__all__ = ["LLMRepository"]

_SYSTEM_AGENTS: list[dict[str, Any]] = [
    {
        "name": "build",
        "agent_type": "system",
        "description": (
            "Системный агент для программирования и генерации кода. "
            "Специализируется на написании, рефакторинге и отладке кода."
        ),
        "system_prompt": (
            "Ты — Build, системный агент для программирования.\n"
            "Твоя задача — писать чистый, эффективный код.\n"
            "Следуй стандартам проекта, добавляй type hints, docstrings.\n"
            "При ошибках — анализируй и исправляй, а не просто переписывай."
        ),
        "model": None,
        "settings": {"temperature": 0.3, "max_tokens": 4096},
    },
    {
        "name": "plan",
        "agent_type": "system",
        "description": (
            "Системный агент для планирования задач и декомпозиции. "
            "Разбивает сложные задачи на подзадачи, определяет зависимости."
        ),
        "system_prompt": (
            "Ты — Plan, системный агент для планирования.\n"
            "Твоя задача — декомпозировать сложные задачи на подзадачи.\n"
            "Определяй зависимости, оценивай трудоёмкость, предлагай порядок выполнения.\n"
            "Формат: нумерованный список с зависимостями."
        ),
        "model": None,
        "settings": {"temperature": 0.5, "max_tokens": 2048},
    },
]


class LLMRepository:
    """Репозиторий для агентов LLM."""

    def __init__(self, pool: Any, log: Any | None = None) -> None:
        self._pool = pool
        self._log = log

    # ── Agents CRUD ─────────────────────────────────────

    async def create_agent(
        self,
        name: str,
        agent_type: str,
        description: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        settings: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        import json
        row = await self._pool.fetchrow(
            "INSERT INTO llm.llm_agents "
            "(name, agent_type, description, system_prompt, model, settings, workspace_id, owner_id) "
            "VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8) "
            "ON CONFLICT (name) DO UPDATE SET "
            "description = EXCLUDED.description, "
            "system_prompt = EXCLUDED.system_prompt, "
            "model = EXCLUDED.model, "
            "settings = EXCLUDED.settings, "
            "updated_at = NOW() "
            "RETURNING *",
            name,
            agent_type,
            description,
            system_prompt,
            model,
            json.dumps(settings or {}),
            workspace_id,
            owner_id,
        )
        return dict(row) if row else {}

    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM llm.llm_agents WHERE id = $1", agent_id,
        )
        return dict(row) if row else None

    async def get_agent_by_name(self, name: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM llm.llm_agents WHERE name = $1", name,
        )
        return dict(row) if row else None

    async def update_agent(
        self, agent_id: str, data: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not data:
            return await self.get_agent(agent_id)

        set_parts = []
        params: list[Any] = []
        idx = 1
        for key, value in data.items():
            if key == "settings" and isinstance(value, dict):
                import json
                set_parts.append(f"{key} = ${idx}::jsonb")
                params.append(json.dumps(value))
            else:
                set_parts.append(f"{key} = ${idx}")
                params.append(value)
            idx += 1

        params.append(agent_id)
        row = await self._pool.fetchrow(
            f"UPDATE llm.llm_agents SET {', '.join(set_parts)}, "
            f"updated_at = NOW() WHERE id = ${idx} RETURNING *",
            *params,
        )
        return dict(row) if row else None

    async def delete_agent(self, agent_id: str) -> bool:
        result = await self._pool.execute(
            "DELETE FROM llm.llm_agents WHERE id = $1", agent_id,
        )
        return "DELETE 1" in str(result)

    async def list_agents(
        self,
        agent_type: str | None = None,
        workspace_id: str | None = None,
        owner_id: str | None = None,
        is_active: bool | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], int]:
        """Список агентов с фильтрами и пагинацией."""
        where_parts = []
        params: list[Any] = []
        idx = 1

        if agent_type:
            where_parts.append(f"agent_type = ${idx}")
            params.append(agent_type)
            idx += 1
        if workspace_id:
            where_parts.append(f"workspace_id = ${idx}")
            params.append(workspace_id)
            idx += 1
        if owner_id:
            where_parts.append(f"owner_id = ${idx}")
            params.append(owner_id)
            idx += 1
        if is_active is not None:
            where_parts.append(f"is_active = ${idx}")
            params.append(is_active)
            idx += 1

        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        total = await self._pool.fetchval(
            f"SELECT COUNT(*) FROM llm.llm_agents {where_clause}",
            *params,
        )

        count_params = list(params)
        count_params.extend([limit, offset])
        rows = await self._pool.fetch(
            f"SELECT * FROM llm.llm_agents {where_clause} "
            f"ORDER BY created_at DESC LIMIT ${idx} OFFSET ${idx + 1}",
            *count_params,
        )
        return [dict(r) for r in rows], total or 0

    async def seed_system_agents(self) -> None:
        """Идемпотентная вставка системных агентов Build и Plan."""
        for spec in _SYSTEM_AGENTS:
            await self.create_agent(**spec)
        if self._log is not None:
            self._log.info("System agents seeded (build, plan)")

    def seed_system_agents_sync(self) -> None:
        """Синхронный сид для on_load. Не через @task."""
        import json

        sql = (
            "INSERT INTO llm.llm_agents "
            "(name, agent_type, description, system_prompt, model, settings) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET "
            "description = EXCLUDED.description, "
            "system_prompt = EXCLUDED.system_prompt, "
            "model = EXCLUDED.model, "
            "settings = EXCLUDED.settings, "
            "updated_at = NOW()"
        )
        if not hasattr(self._pool, "connection"):
            return
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for spec in _SYSTEM_AGENTS:
                    cur.execute(
                        sql,
                        (
                            spec["name"],
                            spec["agent_type"],
                            spec["description"],
                            spec["system_prompt"],
                            spec["model"],
                            json.dumps(spec["settings"] or {}),
                        ),
                    )
        if self._log is not None:
            self._log.info("System agents seeded (build, plan)")

    def _public_provider(self, row: dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        secret = data.pop("api_key", None)
        data["api_key_set"] = bool(secret)
        return data

    async def create_provider(
        self,
        name: str,
        kind: str,
        vendor: str,
        base_url: str | None = None,
        default_model: str | None = None,
        api_key: str | None = None,
        oauth_status: str | None = None,
    ) -> dict[str, Any]:
        row = await self._pool.fetchrow(
            "INSERT INTO llm.llm_providers "
            "(name, kind, vendor, base_url, default_model, api_key, oauth_status) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7) "
            "RETURNING *",
            name,
            kind,
            vendor,
            base_url,
            default_model,
            api_key,
            oauth_status,
        )
        return self._public_provider(dict(row) if row is not None else {})

    async def list_providers(self) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            "SELECT * FROM llm.llm_providers ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            100,
            0,
        )
        return [self._public_provider(dict(row)) for row in rows]

    async def count_agents_by_type(self, agent_type: str) -> int:
        result = await self._pool.fetchval(
            "SELECT COUNT(*) FROM llm.llm_agents WHERE agent_type = $1",
            agent_type,
        )
        return result or 0
