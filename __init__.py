"""LLM Module — модуль LLM для Mia Framework.

Абстракция над провайдерами + управление определениями агентов.

Использование:
    app.load_module("llm")

    provider = app.services.resolve(LLMProvider)
    response = provider.chat(messages=[...])
"""
from __future__ import annotations

import asyncio
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

from argenta_logging import get_logger

log = get_logger(__name__)

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
            dependencies=["log", "db"],
            cache_rules={"get_providers": 300},
            timeout_defaults={"chat": 120.0, "create_agent": 10.0},
        )

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig.from_env()
        self._provider: LLMProvider | None = None

    def on_load(self, state: Any) -> None:
        """Инициализация модуля: провайдеры → БД → AUTH_SCHEMA → DI."""
        self._provider = LLMProvider(self._config)

        try:
            if hasattr(state, "services") and hasattr(state.services, "register"):
                state.services.register(LLMProvider, self._provider)
        except Exception as exc:
            log.warning("failed_to_register_llm_provider", error=str(exc))

        # Инициализация БД + AUTH_SCHEMA (идемпотентно)
        async def _init_llm() -> None:
            await self._provider.initialize(state)

        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_init_llm())
        else:
            loop.run_until_complete(_init_llm())

        log.info(
            "llm_module_loaded",
            version=self.version,
            default_provider=self._config.default_provider,
            providers=list(self._provider.provider_registry.list_providers()),
        )

    def on_unload(self) -> None:
        self._provider = None
        log.info("llm_module_unloaded")
