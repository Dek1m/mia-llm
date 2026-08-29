"""LLM Module — модуль LLM для Mia Framework.

Абстракция над провайдерами + управление определениями агентов.

Использование:
    app.load_module("llm")

    provider = app.services.resolve(LLMProvider)
    response = provider.chat(messages=[...])
"""
from __future__ import annotations

from typing import Any

from modules_system.module_base import ModuleBase, ModuleMeta

from .config import LLMConfig
from .provider import LLMProvider
from .models import AgentDefinition, AgentInfo, ChatMessage, ChatResponse, StreamChunk

__all__ = [
    "LLMModule",
    "LLMProvider",
    "LLMConfig",
    "AgentDefinition",
    "AgentInfo",
    "ChatMessage",
    "ChatResponse",
    "StreamChunk",
]

MODULE_VERSION = "1.0.0"


class LLMModule(ModuleBase):
    """LLM-модуль для Mia Framework.

    Предоставляет:
    - Единый интерфейс chat / chat_stream
    - Управление агентами (CRUD в БД)
    - Провайдеры с fallback
    - Регистрацию AUTH_SCHEMA и DB-схемы
    """

    @property
    def name(self) -> str:
        return "llm"

    @property
    def version(self) -> str:
        return MODULE_VERSION

    @property
    def meta(self) -> ModuleMeta:
        return ModuleMeta(
            dependencies=["log", "db", "workspace"],
            cache_rules={"get_providers": 300},
            timeout_defaults={"chat": 120.0, "create_agent": 10.0},
        )

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig.from_env()
        self._provider: LLMProvider | None = None
        self._log = None

    def on_load(self, state: Any) -> None:
        """Инициализация модуля: провайдеры → БД → AUTH_SCHEMA → DI."""
        self._log = state.log
        self._provider = LLMProvider(self._config, log=self._log)

        try:
            if hasattr(state, "services") and hasattr(state.services, "register"):
                state.services.register(LLMProvider, self._provider)
            from modules.db.provider import DatabaseProvider

            database = state.services.resolve(DatabaseProvider)
            self._provider.bind_runtime(state, database)
            from modules.auth.provider import AuthProvider
            from .schema import LLM_SCHEMA

            auth = state.services.resolve(AuthProvider)
            if auth.registry is not None:
                auth.registry.register_sync("llm", LLM_SCHEMA, is_builtin=False)
        except Exception as exc:
            self._log.warning(
                "failed_to_register_llm_provider",
                extra={"error": str(exc)},
            )
        state.llm = self._provider

        self._log.info(
            "llm_module_loaded",
            extra={
                "version": self.version,
                "default_provider": self._config.default_provider,
                "providers": list(self._provider.provider_registry.list_providers()),
            },
        )

    def apply_schema(self, state: Any) -> None:
        """DDL llm.* + auth seed агентов. Только migrate."""
        if self._provider is not None:
            self._provider.initialize_sync(state)

    def on_unload(self) -> None:
        self._provider = None
        self._log.info("llm_module_unloaded")
        self._log = None
