"""LLM Provider — основной провайдер модуля LLM.

Интеграция с БД (агенты), провайдерами (chat), auth (permissions).
"""
from __future__ import annotations

from typing import Any


from core.task_decorator import task

from .config import LLMConfig
from .repository import LLMRepository
from .schema import LLM_SCHEMA
from .schemas import DB_SCHEMA
from .models import AgentInfo
from .providers.base import BaseProvider
from .providers.openai import OpenAIProvider
from .providers.registry import ProviderRegistry


__all__ = ["LLMProvider"]


class LLMError(Exception):
    """Базовая ошибка LLM-модуля."""

    def __init__(self, message: str, code: str = "LLM_ERROR") -> None:
        self.code = code
        super().__init__(message)


class NotFoundError(LLMError):
    def __init__(self, entity: str = "Resource") -> None:
        super().__init__(f"{entity} not found", "NOT_FOUND")


class ForbiddenError(LLMError):
    def __init__(self, message: str = "Forbidden") -> None:
        super().__init__(message, "FORBIDDEN")


class LLMProvider:
    """Провайдер LLM.

    Предоставляет:
    - Вызов LLM через провайдеры с fallback
    - Управление агентами (CRUD в БД)
    - Просмотр провайдеров и моделей
    """

    def __init__(self, config: LLMConfig, log: Any = None) -> None:
        self._config = config
        self._log = log
        self._repo: LLMRepository | None = None
        self._provider_registry = ProviderRegistry(log=log)
        self._init_providers()

    def _init_providers(self) -> None:
        """Создать и зарегистрировать провайдеры из конфига."""
        for name, pcfg in self._config.providers.items():
            provider = OpenAIProvider(
                name=name,
                base_url=pcfg.base_url,
                api_key=pcfg.api_key,
                default_model=pcfg.default_model,
                timeout=pcfg.timeout,
                log=self._log,
            )
            self._provider_registry.register(name, provider)

        if self._config.default_provider:
            self._provider_registry.set_default(self._config.default_provider)
        if self._config.fallback_provider:
            self._provider_registry.set_fallback(self._config.fallback_provider)

    @property
    def repository(self) -> LLMRepository | None:
        return self._repo

    @property
    def provider_registry(self) -> ProviderRegistry:
        return self._provider_registry

    @task(type="database")
    async def initialize(self, state: Any) -> None:
        """Регистрация БД-схемы и AUTH_SCHEMA."""
        self.initialize_sync(state)

    def initialize_sync(self, state: Any) -> None:
        """Синхронная версия initialize для on_load. Не через @task."""
        from modules.db.provider import DatabaseProvider

        db_provider = state.services.resolve(DatabaseProvider)
        self._repo = LLMRepository(db_provider.pool, log=self._log)
        db_provider.register_schema(
            "llm",
            DB_SCHEMA,
            schema_name="llm",
            ddl_dir="ddl",
        )
        from modules.auth.schema_registry import AuthSchemaRegistry

        auth_registry = state.services.resolve(AuthSchemaRegistry)
        auth_registry.register_sync("llm", LLM_SCHEMA, is_builtin=False)
        self._repo.seed_system_agents_sync()
        self._log.info("LLM schema registered, system agents seeded")

    # ── Chat ────────────────────────────────────────────

    @task(
        type="network",
        api=True,
        permission="llm:chat",
        name="chat",
        description="Вызов LLM через chat completions",
        args={"messages": "list", "model": "str", "provider": "str"},
        return_type="dict",
    )
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Вызвать LLM через провайдер с fallback."""
        return await self._provider_registry.chat_with_fallback(
            messages=messages,
            model=model,
            **kwargs,
        )

    @task(
        type="network",
        api=True,
        permission="llm:chat_stream",
        name="chat_stream",
        description="Потоковый вывод LLM (пока синхронный обёртка)",
        args={"messages": "list", "model": "str"},
        return_type="dict",
    )
    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Потоковый chat (заготовка — в первой итерации просто chat)."""
        return await self.chat(messages=messages, model=model, **kwargs)

    # ── Agents ──────────────────────────────────────────

    @task(
        type="database",
        api=True,
        permission="llm:agent_list",
        name="agents",
        description="Список всех агентов (системные + пользовательские + workspace)",
        args={"agent_type": "str", "workspace_id": "str", "offset": "int", "limit": "int"},
        return_type="dict",
    )
    async def agents(
        self,
        agent_type: str | None = None,
        workspace_id: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Получить список агентов."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")
        items, total = await self._repo.list_agents(
            agent_type=agent_type, workspace_id=workspace_id,
            offset=offset, limit=limit,
        )
        return {"items": items, "total": total, "offset": offset, "limit": limit}

    @task(
        type="database",
        api=True,
        permission="llm:agent_list",
        name="agent",
        description="Получить информацию об агенте",
        args={"agent_id": "str"},
        return_type="dict",
    )
    async def agent(self, agent_id: str) -> dict[str, Any]:
        """Получить агента по ID."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")
        row = await self._repo.get_agent(agent_id)
        if not row:
            raise NotFoundError("Agent")
        return row

    @task(
        type="database",
        api=True,
        permission="llm:agent_manage",
        name="create_agent",
        description="Создать нового агента (пользовательский/workspace)",
        args={
            "name": "str", "agent_type": "str", "description": "str",
            "system_prompt": "str", "model": "str", "workspace_id": "str",
        },
        return_type="dict",
    )
    async def create_agent(
        self,
        name: str,
        agent_type: str = "user",
        description: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        workspace_id: str | None = None,
        owner_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Создать агента."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")

        if agent_type == "system":
            raise ForbiddenError("Cannot create system agents manually")

        row = await self._repo.create_agent(
            name=name,
            agent_type=agent_type,
            description=description,
            system_prompt=system_prompt,
            model=model,
            workspace_id=workspace_id,
            owner_id=owner_id,
        )
        return row

    @task(
        type="database",
        api=True,
        permission="llm:agent_manage",
        name="update_agent",
        description="Обновить агента",
        args={"agent_id": "str", "data": "dict"},
        return_type="dict",
    )
    async def update_agent(self, agent_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Обновить агента."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")

        # Проверяем, не system ли агент
        row = await self._repo.get_agent(agent_id)
        if not row:
            raise NotFoundError("Agent")
        if row.get("agent_type") == "system":
            raise ForbiddenError("Cannot modify system agents")

        result = await self._repo.update_agent(agent_id, data)
        if not result:
            raise NotFoundError("Agent")
        return result

    @task(
        type="database",
        api=True,
        permission="llm:agent_manage",
        name="delete_agent",
        description="Удалить агента",
        args={"agent_id": "str"},
        return_type="bool",
    )
    async def delete_agent(self, agent_id: str) -> bool:
        """Удалить агента."""
        if self._repo is None:
            raise LLMError("LLM not initialized (no DB pool)")

        row = await self._repo.get_agent(agent_id)
        if not row:
            raise NotFoundError("Agent")
        if row.get("agent_type") == "system":
            raise ForbiddenError("Cannot delete system agents")

        return await self._repo.delete_agent(agent_id)

    # ── Providers ───────────────────────────────────────

    @task(
        type="cpu",
        api=True,
        permission="llm:config",
        name="get_providers",
        description="Список зарегистрированных LLM-провайдеров и их статус",
        args={},
        return_type="list",
    )
    async def get_providers(self) -> list[dict[str, Any]]:
        """Получить список провайдеров с health-check."""
        providers = self._provider_registry.list_providers()
        for p in providers:
            prov = self._provider_registry.get(p["name"])
            if prov:
                try:
                    p["healthy"] = await prov.health()
                except Exception:
                    p["healthy"] = False
            else:
                p["healthy"] = False
        return providers
