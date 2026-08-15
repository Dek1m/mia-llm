"""LLM Module — модуль LLM для Mia Framework.

Абстракция над провайдерами + agent definitions.

Использование:
    app.load_module("llm")

    provider = app.services.resolve(LLMProvider)
    response = provider.chat(messages=[...])
"""
from __future__ import annotations

from typing import Any

from modules_system.module_base import ModuleBase

from .config import LLMConfig
from .provider import LLMProvider
from .models import AgentDefinition, ChatMessage, ChatResponse, StreamChunk

__all__ = [
    "LLMModule",
    "LLMProvider",
    "LLMConfig",
    "AgentDefinition",
    "ChatMessage",
    "ChatResponse",
    "StreamChunk",
]

from argenta_logging import get_logger

log = get_logger(__name__)

MODULE_VERSION = "0.1.0"


class LLMModule(ModuleBase):
    """LLM-модуль для Mia Framework.

    Предоставляет:
    - Единый интерфейс chat / chat_stream
    - Регистрацию и использование agent definitions
    - Интеграцию с Task System
    """

    @property
    def name(self) -> str:
        return "llm"

    @property
    def version(self) -> str:
        return MODULE_VERSION

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig.from_env()
        self._provider: LLMProvider | None = None

    def on_load(self, state: Any) -> None:
        self._provider = LLMProvider(self._config)

        try:
            if hasattr(state, "services") and hasattr(state.services, "register"):
                state.services.register(LLMProvider, self._provider)
                log.info("LLMProvider registered in DI")
        except Exception as exc:
            log.warning("failed_to_register_llm_provider", error=str(exc))

        log.info(
            "llm_module_loaded",
            version=self.version,
            provider=self._config.default_provider,
            model=self._config.default_model,
        )

    def on_unload(self) -> None:
        self._provider = None
        log.info("llm_module_unloaded")
