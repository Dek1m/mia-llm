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
        args = (
            name,
            agent_type,
            description,
            system_prompt,
            model,
            json.dumps(settings or {}),
            workspace_id,
            owner_id,
        )
        if self._psycopg_pool():
            row = self._fetch_one(
                "INSERT INTO llm.llm_agents "
                "(name, agent_type, description, system_prompt, model, settings, workspace_id, owner_id) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s) "
                "ON CONFLICT (name) DO UPDATE SET "
                "description = EXCLUDED.description, "
                "system_prompt = EXCLUDED.system_prompt, "
                "model = EXCLUDED.model, "
                "settings = EXCLUDED.settings, "
                "updated_at = NOW() "
                "RETURNING *",
                args,
            )
            return self._public_agent(row)
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
            *args,
        )
        return self._public_agent(dict(row) if row else {})

    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        if self._psycopg_pool():
            row = self._fetch_one(
                "SELECT * FROM llm.llm_agents WHERE id = %s", (agent_id,),
            )
            if not row:
                return None
            stamped = await self._stamp_avatars([row])
            return self._public_agent(stamped[0])
        row = await self._pool.fetchrow(
            "SELECT * FROM llm.llm_agents WHERE id = $1", agent_id,
        )
        if not row:
            return None
        stamped = await self._stamp_avatars([dict(row)])
        return self._public_agent(stamped[0])

    async def get_agent_by_name(self, name: str) -> dict[str, Any] | None:
        if self._psycopg_pool():
            row = self._fetch_one(
                "SELECT * FROM llm.llm_agents WHERE name = %s", (name,),
            )
            return self._public_agent(row) or None
        row = await self._pool.fetchrow(
            "SELECT * FROM llm.llm_agents WHERE name = $1", name,
        )
        return self._public_agent(dict(row)) if row else None

    async def clear_agent_defaults(self) -> None:
        if self._psycopg_pool():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE llm.llm_agents SET is_default = FALSE WHERE is_default = TRUE")
            return
        await self._pool.execute("UPDATE llm.llm_agents SET is_default = FALSE WHERE is_default = TRUE")

    async def update_agent(
        self, agent_id: str, data: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not data:
            return await self.get_agent(agent_id)

        allowed = {
            "name", "agent_type", "description", "system_prompt", "model",
            "settings", "workspace_id", "owner_id", "is_active",
            "is_visible", "is_default",
        }
        set_parts = []
        params: list[Any] = []
        idx = 1
        for key, value in data.items():
            if key not in allowed:
                continue
            if key == "settings" and isinstance(value, dict):
                import json
                if self._psycopg_pool():
                    set_parts.append(f"{key} = %s::jsonb")
                else:
                    set_parts.append(f"{key} = ${idx}::jsonb")
                params.append(json.dumps(value))
            elif self._psycopg_pool():
                set_parts.append(f"{key} = %s")
                params.append(value)
            else:
                set_parts.append(f"{key} = ${idx}")
                params.append(value)
            idx += 1
        if not set_parts:
            return await self.get_agent(agent_id)

        params.append(agent_id)
        if self._psycopg_pool():
            row = self._fetch_one(
                f"UPDATE llm.llm_agents SET {', '.join(set_parts)}, "
                "updated_at = NOW() WHERE id = %s RETURNING *",
                tuple(params),
            )
            return self._public_agent(row) or None
        row = await self._pool.fetchrow(
            f"UPDATE llm.llm_agents SET {', '.join(set_parts)}, "
            f"updated_at = NOW() WHERE id = ${idx} RETURNING *",
            *params,
        )
        return self._public_agent(dict(row)) if row else None

    async def delete_agent(self, agent_id: str) -> bool:
        if self._psycopg_pool():
            result = self._execute(
                "DELETE FROM llm.llm_agents WHERE id = %s", (agent_id,),
            )
            return "DELETE 1" in str(result)
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
        pg = self._psycopg_pool()
        where_parts = []
        params: list[Any] = []
        idx = 1

        def mark() -> str:
            nonlocal idx
            if pg:
                return "%s"
            token = f"${idx}"
            idx += 1
            return token

        if agent_type:
            where_parts.append(f"agent_type = {mark()}")
            params.append(agent_type)
        if workspace_id:
            where_parts.append(f"workspace_id = {mark()}")
            params.append(workspace_id)
        if owner_id:
            where_parts.append(f"owner_id = {mark()}")
            params.append(owner_id)
        if is_active is not None:
            where_parts.append(f"is_active = {mark()}")
            params.append(is_active)

        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        if pg:
            total = self._fetch_val(
                f"SELECT COUNT(*) FROM llm.llm_agents {where_clause}", tuple(params),
            )
            rows = self._fetch_all(
                f"SELECT * FROM llm.llm_agents {where_clause} "
                "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                tuple([*params, limit, offset]),
            )
            return [self._public_agent(row) for row in await self._stamp_avatars(rows)], total or 0

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
        return [self._public_agent(row) for row in await self._stamp_avatars([dict(r) for r in rows])], total or 0

    async def _stamp_avatars(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ids = [str(row.get("id") or "") for row in rows if row.get("id")]
        marked: set[str] = set()
        if ids and self._psycopg_pool():
            found = self._fetch_all(
                "SELECT agent_id FROM llm.llm_agent_avatars WHERE agent_id = ANY(%s)",
                (ids,),
            )
            marked = {str(item["agent_id"]) for item in found}
        for row in rows:
            row["has_avatar"] = str(row.get("id") or "") in marked
        return rows

    async def get_agent_avatar(self, agent_id: str) -> dict[str, Any] | None:
        if self._psycopg_pool():
            row = self._fetch_one(
                "SELECT bytes, content_type FROM llm.llm_agent_avatars WHERE agent_id = %s",
                (agent_id,),
            )
            return row or None
        row = await self._pool.fetchrow(
            "SELECT bytes, content_type FROM llm.llm_agent_avatars WHERE agent_id = $1",
            agent_id,
        )
        return dict(row) if row else None

    async def upsert_agent_avatar(self, agent_id: str, data: bytes, content_type: str) -> None:
        if self._psycopg_pool():
            existing = self._fetch_one(
                "SELECT agent_id FROM llm.llm_agent_avatars WHERE agent_id = %s",
                (agent_id,),
            )
            if existing:
                self._execute(
                    "UPDATE llm.llm_agent_avatars SET bytes = %s, content_type = %s, updated_at = NOW() "
                    "WHERE agent_id = %s",
                    (data, content_type, agent_id),
                )
                return
            self._execute(
                "INSERT INTO llm.llm_agent_avatars (agent_id, bytes, content_type) VALUES (%s, %s, %s)",
                (agent_id, data, content_type),
            )
            return
        await self._pool.execute(
            "INSERT INTO llm.llm_agent_avatars (agent_id, bytes, content_type) "
            "VALUES ($1, $2, $3) ON CONFLICT (agent_id) DO UPDATE SET "
            "bytes = EXCLUDED.bytes, content_type = EXCLUDED.content_type, updated_at = NOW()",
            agent_id, data, content_type,
        )

    async def delete_agent_avatar(self, agent_id: str) -> None:
        if self._psycopg_pool():
            self._execute(
                "DELETE FROM llm.llm_agent_avatars WHERE agent_id = %s", (agent_id,),
            )
            return
        await self._pool.execute(
            "DELETE FROM llm.llm_agent_avatars WHERE agent_id = $1", agent_id,
        )

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

    def _public_agent(self, row: dict[str, Any] | None) -> dict[str, Any]:
        if not row:
            return {}
        data = dict(row)
        for key in ("id", "workspace_id", "owner_id"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        for key in ("created_at", "updated_at"):
            value = data.get(key)
            iso = getattr(value, "isoformat", None)
            if callable(iso):
                data[key] = iso()
        agent_id = data.get("id")
        if data.pop("has_avatar", False) and agent_id:
            data["avatar_url"] = f"/api/v1/llm/agent_avatar?agent_id={agent_id}"
        else:
            data["avatar_url"] = None
        return data

    def _public_provider(self, row: dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        secret = data.pop("api_key", None)
        data["api_key_set"] = bool(secret)
        return data

    def _psycopg_pool(self) -> bool:
        return hasattr(self._pool, "connection")

    def _fetch_all(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                return [dict(zip(columns, item)) for item in cur.fetchall()]

    def _fetch_one(self, sql: str, args: tuple[Any, ...]) -> dict[str, Any]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                raw = cur.fetchone()
                columns = [desc[0] for desc in cur.description] if cur.description else []
                return dict(zip(columns, raw)) if raw else {}

    def _fetch_val(self, sql: str, args: tuple[Any, ...] = ()) -> Any:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                raw = cur.fetchone()
                return raw[0] if raw else None

    def _execute(self, sql: str, args: tuple[Any, ...] = ()) -> str:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                return cur.statusmessage or "OK"

    async def create_provider(
        self,
        name: str,
        kind: str,
        vendor: str,
        description: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        api_key: str | None = None,
        oauth_status: str | None = None,
        models: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        args = (name, kind, vendor, description, base_url, default_model, api_key, oauth_status)
        if self._psycopg_pool():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO llm.llm_providers "
                        "(name, kind, vendor, description, base_url, default_model, api_key, oauth_status) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                        args,
                    )
                    raw = cur.fetchone()
                    columns = [desc[0] for desc in cur.description]
                    row = dict(zip(columns, raw)) if raw else {}
                    provider_id = row.get("id")
                    if provider_id and models:
                        for item in models:
                            cur.execute(
                                "INSERT INTO llm.llm_models "
                                "(provider_id, model_id, display_name, enabled, is_available, "
                                "supports_reasoning, reasoning_enabled, reasoning_effort) "
                                "VALUES (%s, %s, %s, %s, TRUE, %s, %s, %s) "
                                "ON CONFLICT (provider_id, model_id) DO UPDATE SET "
                                "display_name = EXCLUDED.display_name, "
                                "enabled = EXCLUDED.enabled, "
                                "supports_reasoning = EXCLUDED.supports_reasoning, "
                                "reasoning_enabled = EXCLUDED.reasoning_enabled, "
                                "reasoning_effort = EXCLUDED.reasoning_effort, "
                                "is_available = TRUE, updated_at = NOW()",
                                (
                                    provider_id,
                                    item["model_id"],
                                    item.get("display_name") or item["model_id"],
                                    bool(item.get("enabled", True)),
                                    bool(item.get("supports_reasoning", False)),
                                    bool(item.get("reasoning_enabled", False)),
                                    item.get("reasoning_effort"),
                                ),
                            )
            public = self._public_provider(row)
            public["models"] = await self.list_models(str(row.get("id")))
            return public
        row = await self._pool.fetchrow(
            "INSERT INTO llm.llm_providers "
            "(name, kind, vendor, description, base_url, default_model, api_key, oauth_status) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
            "RETURNING *",
            *args,
        )
        data = dict(row) if row is not None else {}
        provider_id = data.get("id")
        if provider_id and models:
            for item in models:
                await self._pool.fetchrow(
                    "INSERT INTO llm.llm_models "
                    "(provider_id, model_id, display_name, enabled, is_available, "
                    "supports_reasoning, reasoning_enabled, reasoning_effort) "
                    "VALUES ($1, $2, $3, $4, TRUE, $5, $6, $7) RETURNING *",
                    provider_id,
                    item["model_id"],
                    item.get("display_name") or item["model_id"],
                    bool(item.get("enabled", True)),
                    bool(item.get("supports_reasoning", False)),
                    bool(item.get("reasoning_enabled", False)),
                    item.get("reasoning_effort"),
                )
        public = self._public_provider(data)
        public["models"] = await self.list_models(str(provider_id)) if provider_id else []
        return public

    async def list_models(self, provider_id: str) -> list[dict[str, Any]]:
        if self._psycopg_pool():
            return self._fetch_all(
                "SELECT id, provider_id, model_id, display_name, enabled, is_available, "
                "supports_reasoning, reasoning_enabled, reasoning_effort "
                "FROM llm.llm_models WHERE provider_id = %s ORDER BY display_name",
                (provider_id,),
            )
        rows = await self._pool.fetch(
            "SELECT id, provider_id, model_id, display_name, enabled, is_available, "
            "supports_reasoning, reasoning_enabled, reasoning_effort "
            "FROM llm.llm_models WHERE provider_id = $1 ORDER BY display_name",
            provider_id,
        )
        return [dict(row) for row in rows]

    async def list_providers(self) -> list[dict[str, Any]]:
        if self._psycopg_pool():
            rows = self._fetch_all(
                "SELECT * FROM llm.llm_providers ORDER BY created_at DESC LIMIT 100"
            )
        else:
            fetched = await self._pool.fetch(
                "SELECT * FROM llm.llm_providers ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                100,
                0,
            )
            rows = [dict(row) for row in fetched]
        items = [self._public_provider(row) for row in rows]
        for item in items:
            item["models"] = await self.list_models(str(item["id"]))
        return items

    async def list_providers_with_secrets(self) -> list[dict[str, Any]]:
        if self._psycopg_pool():
            rows = self._fetch_all(
                "SELECT * FROM llm.llm_providers ORDER BY created_at DESC LIMIT 100"
            )
        else:
            fetched = await self._pool.fetch(
                "SELECT * FROM llm.llm_providers ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                100,
                0,
            )
            rows = [dict(row) for row in fetched]
        return rows

    async def get_provider(self, provider_id: str) -> dict[str, Any]:
        if self._psycopg_pool():
            return self._fetch_one(
                "SELECT * FROM llm.llm_providers WHERE id = %s",
                (provider_id,),
            )
        row = await self._pool.fetchrow(
            "SELECT * FROM llm.llm_providers WHERE id = $1",
            provider_id,
        )
        return dict(row) if row is not None else {}

    async def set_oauth_state(
        self,
        provider_id: str,
        api_key: str,
        oauth_status: str,
    ) -> dict[str, Any]:
        if self._psycopg_pool():
            row = self._fetch_one(
                "UPDATE llm.llm_providers SET api_key = %s, oauth_status = %s, updated_at = NOW() "
                "WHERE id = %s RETURNING *",
                (api_key, oauth_status, provider_id),
            )
        else:
            fetched = await self._pool.fetchrow(
                "UPDATE llm.llm_providers SET api_key = $1, oauth_status = $2, updated_at = NOW() "
                "WHERE id = $3 RETURNING *",
                api_key,
                oauth_status,
                provider_id,
            )
            row = dict(fetched) if fetched is not None else {}
        if not row:
            return {}
        public = self._public_provider(row)
        public["models"] = await self.list_models(provider_id)
        return public

    async def replace_remote_models(
        self,
        provider_id: str,
        remote_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Сверить каталог. Возвращает enabled-модели, которых больше нет у вендора."""
        vanished: list[dict[str, Any]] = []
        stored = await self.list_models(provider_id)
        remote = set(remote_ids)
        if self._psycopg_pool():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    for item in stored:
                        if item["model_id"] not in remote:
                            if item.get("enabled"):
                                vanished.append(item)
                            cur.execute(
                                "UPDATE llm.llm_models SET is_available = FALSE, updated_at = NOW() "
                                "WHERE id = %s",
                                (item["id"],),
                            )
                        else:
                            cur.execute(
                                "UPDATE llm.llm_models SET is_available = TRUE, updated_at = NOW() "
                                "WHERE id = %s",
                                (item["id"],),
                            )
        else:
            for item in stored:
                if item["model_id"] not in remote and item.get("enabled"):
                    vanished.append(item)
        return vanished

    async def set_model_enabled(self, model_uuid: str, enabled: bool) -> dict[str, Any]:
        if self._psycopg_pool():
            row = self._fetch_one(
                "UPDATE llm.llm_models SET enabled = %s, updated_at = NOW() "
                "WHERE id = %s RETURNING id, provider_id, model_id, display_name, enabled, is_available, "
                "supports_reasoning, reasoning_enabled, reasoning_effort",
                (enabled, model_uuid),
            )
            return row
        row = await self._pool.fetchrow(
            "UPDATE llm.llm_models SET enabled = $1, updated_at = NOW() "
            "WHERE id = $2 RETURNING id, provider_id, model_id, display_name, enabled, is_available, "
            "supports_reasoning, reasoning_enabled, reasoning_effort",
            enabled,
            model_uuid,
        )
        return dict(row) if row is not None else {}

    async def set_model_reasoning(
        self,
        model_uuid: str,
        reasoning_enabled: bool,
        reasoning_effort: str,
    ) -> dict[str, Any]:
        if self._psycopg_pool():
            return self._fetch_one(
                "UPDATE llm.llm_models SET reasoning_enabled = %s, reasoning_effort = %s, "
                "updated_at = NOW() WHERE id = %s AND supports_reasoning = TRUE "
                "RETURNING id, provider_id, model_id, display_name, enabled, is_available, "
                "supports_reasoning, reasoning_enabled, reasoning_effort",
                (reasoning_enabled, reasoning_effort, model_uuid),
            )
        row = await self._pool.fetchrow(
            "UPDATE llm.llm_models SET reasoning_enabled = $1, reasoning_effort = $2, "
            "updated_at = NOW() WHERE id = $3 AND supports_reasoning = TRUE "
            "RETURNING id, provider_id, model_id, display_name, enabled, is_available, "
            "supports_reasoning, reasoning_enabled, reasoning_effort",
            reasoning_enabled,
            reasoning_effort,
            model_uuid,
        )
        return dict(row) if row is not None else {}

    async def update_provider(
        self,
        provider_id: str,
        name: str,
        description: str | None,
        base_url: str | None,
        api_key: str | None = None,
        models: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._psycopg_pool():
            if api_key:
                row = self._fetch_one(
                    "UPDATE llm.llm_providers SET name = %s, description = %s, base_url = %s, "
                    "api_key = %s, updated_at = NOW() WHERE id = %s RETURNING *",
                    (name, description, base_url, api_key, provider_id),
                )
            else:
                row = self._fetch_one(
                    "UPDATE llm.llm_providers SET name = %s, description = %s, base_url = %s, "
                    "updated_at = NOW() WHERE id = %s RETURNING *",
                    (name, description, base_url, provider_id),
                )
            if not row:
                return {}
            if models:
                await self.upsert_models(provider_id, models)
            public = self._public_provider(row)
            public["models"] = await self.list_models(provider_id)
            return public
        if api_key:
            row = await self._pool.fetchrow(
                "UPDATE llm.llm_providers SET name = $1, description = $2, base_url = $3, "
                "api_key = $4, updated_at = NOW() WHERE id = $5 RETURNING *",
                name,
                description,
                base_url,
                api_key,
                provider_id,
            )
        else:
            row = await self._pool.fetchrow(
                "UPDATE llm.llm_providers SET name = $1, description = $2, base_url = $3, "
                "updated_at = NOW() WHERE id = $4 RETURNING *",
                name,
                description,
                base_url,
                provider_id,
            )
        data = dict(row) if row is not None else {}
        if not data:
            return {}
        if models:
            await self.upsert_models(provider_id, models)
        public = self._public_provider(data)
        public["models"] = await self.list_models(provider_id)
        return public

    async def upsert_models(self, provider_id: str, models: list[dict[str, Any]]) -> None:
        if not models:
            return
        if self._psycopg_pool():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    for item in models:
                        cur.execute(
                            "INSERT INTO llm.llm_models "
                            "(provider_id, model_id, display_name, enabled, is_available, "
                            "supports_reasoning, reasoning_enabled, reasoning_effort) "
                            "VALUES (%s, %s, %s, %s, TRUE, %s, %s, %s) "
                            "ON CONFLICT (provider_id, model_id) DO UPDATE SET "
                            "display_name = EXCLUDED.display_name, "
                            "enabled = EXCLUDED.enabled, "
                            "supports_reasoning = EXCLUDED.supports_reasoning, "
                            "reasoning_enabled = EXCLUDED.reasoning_enabled, "
                            "reasoning_effort = EXCLUDED.reasoning_effort, "
                            "is_available = TRUE, updated_at = NOW()",
                            (
                                provider_id,
                                item["model_id"],
                                item.get("display_name") or item["model_id"],
                                bool(item.get("enabled", True)),
                                bool(item.get("supports_reasoning", False)),
                                bool(item.get("reasoning_enabled", False)),
                                item.get("reasoning_effort"),
                            ),
                        )
            return
        for item in models:
            await self._pool.fetchrow(
                "INSERT INTO llm.llm_models "
                "(provider_id, model_id, display_name, enabled, is_available, "
                "supports_reasoning, reasoning_enabled, reasoning_effort) "
                "VALUES ($1, $2, $3, $4, TRUE, $5, $6, $7) RETURNING *",
                provider_id,
                item["model_id"],
                item.get("display_name") or item["model_id"],
                bool(item.get("enabled", True)),
                bool(item.get("supports_reasoning", False)),
                bool(item.get("reasoning_enabled", False)),
                item.get("reasoning_effort"),
            )

    def reencrypt_api_keys_sync(self) -> int:
        from .secrets import encrypt_secret, is_encrypted

        if not self._psycopg_pool():
            return 0
        rows = self._fetch_all(
            "SELECT id, api_key FROM llm.llm_providers WHERE api_key IS NOT NULL AND api_key <> ''"
        )
        n = 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    stored = str(row.get("api_key") or "")
                    if is_encrypted(stored):
                        continue
                    cur.execute(
                        "UPDATE llm.llm_providers SET api_key = %s WHERE id = %s",
                        (encrypt_secret(stored), row["id"]),
                    )
                    n += 1
        return n

    async def set_model_name(self, model_uuid: str, display_name: str) -> dict[str, Any]:
        name = display_name.strip()
        if self._psycopg_pool():
            return self._fetch_one(
                "UPDATE llm.llm_models SET display_name = %s, updated_at = NOW() "
                "WHERE id = %s RETURNING id, provider_id, model_id, display_name, enabled, is_available, "
                "supports_reasoning, reasoning_enabled, reasoning_effort",
                (name, model_uuid),
            )
        row = await self._pool.fetchrow(
            "UPDATE llm.llm_models SET display_name = $1, updated_at = NOW() "
            "WHERE id = $2 RETURNING id, provider_id, model_id, display_name, enabled, is_available, "
            "supports_reasoning, reasoning_enabled, reasoning_effort",
            name,
            model_uuid,
        )
        return dict(row) if row is not None else {}

    async def set_provider_models_enabled(self, provider_id: str, enabled: bool) -> int:
        if self._psycopg_pool():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE llm.llm_models SET enabled = %s, updated_at = NOW() "
                        "WHERE provider_id = %s",
                        (enabled, provider_id),
                    )
                    return cur.rowcount
        result = await self._pool.execute(
            "UPDATE llm.llm_models SET enabled = $1, updated_at = NOW() WHERE provider_id = $2",
            enabled,
            provider_id,
        )
        text = str(result or "")
        parts = text.split()
        return int(parts[-1]) if parts and parts[-1].isdigit() else 0

    async def delete_model(self, model_id: str) -> bool:
        if self._psycopg_pool():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM llm.llm_models WHERE id = %s", (model_id,))
                    return cur.rowcount > 0
        result = await self._pool.execute("DELETE FROM llm.llm_models WHERE id = $1", model_id)
        return bool(result) and "DELETE 0" not in str(result)

    async def delete_provider(self, provider_id: str) -> bool:
        if self._psycopg_pool():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM llm.llm_providers WHERE id = %s", (provider_id,))
                    return cur.rowcount > 0
        result = await self._pool.execute(
            "DELETE FROM llm.llm_providers WHERE id = $1",
            provider_id,
        )
        return bool(result) and "DELETE 0" not in str(result)

    async def count_agents_by_type(self, agent_type: str) -> int:
        result = await self._pool.fetchval(
            "SELECT COUNT(*) FROM llm.llm_agents WHERE agent_type = $1",
            agent_type,
        )
        return result or 0

    async def everyone_group_id(self) -> str | None:
        if self._psycopg_pool():
            row = self._fetch_one(
                "SELECT id FROM auth.groups WHERE name = %s AND is_builtin = TRUE",
                ("Everyone",),
            )
            value = row.get("id")
            return str(value) if value else None
        row = await self._pool.fetchrow(
            "SELECT id FROM auth.groups WHERE name = $1 AND is_builtin = TRUE",
            "Everyone",
        )
        if not row:
            return None
        value = dict(row).get("id")
        return str(value) if value else None

    async def insert_share(self, owner_id: str, provider_id: str, group_id: str) -> dict[str, Any]:
        if self._psycopg_pool():
            return self._fetch_one(
                "INSERT INTO llm.provider_shares (owner_id, provider_id, group_id) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (owner_id, provider_id, group_id) DO UPDATE SET owner_id = EXCLUDED.owner_id "
                "RETURNING *",
                (owner_id, provider_id, group_id),
            )
        row = await self._pool.fetchrow(
            "INSERT INTO llm.provider_shares (owner_id, provider_id, group_id) "
            "VALUES ($1, $2, $3) "
            "ON CONFLICT (owner_id, provider_id, group_id) DO UPDATE SET owner_id = EXCLUDED.owner_id "
            "RETURNING *",
            owner_id,
            provider_id,
            group_id,
        )
        return dict(row) if row is not None else {}

    async def delete_share(self, provider_id: str, group_id: str) -> bool:
        if self._psycopg_pool():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM llm.provider_shares WHERE provider_id = %s AND group_id = %s",
                        (provider_id, group_id),
                    )
                    return cur.rowcount > 0
        result = await self._pool.execute(
            "DELETE FROM llm.provider_shares WHERE provider_id = $1 AND group_id = $2",
            provider_id,
            group_id,
        )
        return bool(result) and "DELETE 0" not in str(result)

    async def list_shares_for_provider(self, provider_id: str) -> list[dict[str, Any]]:
        if self._psycopg_pool():
            return self._fetch_all(
                "SELECT * FROM llm.provider_shares WHERE provider_id = %s",
                (provider_id,),
            )
        rows = await self._pool.fetch(
            "SELECT * FROM llm.provider_shares WHERE provider_id = $1",
            provider_id,
        )
        return [dict(row) for row in rows]

    async def delete_shares_for_provider(self, owner_id: str, provider_id: str) -> None:
        if self._psycopg_pool():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM llm.provider_shares WHERE owner_id = %s AND provider_id = %s",
                        (owner_id, provider_id),
                    )
            return
        await self._pool.execute(
            "DELETE FROM llm.provider_shares WHERE owner_id = $1 AND provider_id = $2",
            owner_id,
            provider_id,
        )

    async def list_shares_for_groups(self, group_ids: list[str]) -> list[dict[str, Any]]:
        if not group_ids:
            return []
        if self._psycopg_pool():
            return self._fetch_all(
                "SELECT * FROM llm.provider_shares WHERE group_id = ANY(%s)",
                (group_ids,),
            )
        rows = await self._pool.fetch(
            "SELECT * FROM llm.provider_shares WHERE group_id = ANY($1::uuid[])",
            group_ids,
        )
        return [dict(row) for row in rows]
