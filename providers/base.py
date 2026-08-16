"""Base LLM Provider — абстракция для всех LLM-бэкендов."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from argenta_logging import get_logger

log = get_logger(__name__)

__all__ = ["BaseProvider"]


class BaseProvider(ABC):
    """Абстрактный базовый класс для LLM-провайдеров.

    Все провайдеры (OpenAI, llama.cpp и т.д.) наследуют этот класс.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Выполнить chat completion.

        Args:
            messages: Список сообщений [{role, content}, ...].
            model: Модель (по умолчанию из конфига).
            temperature: Температура генерации.
            max_tokens: Максимум токенов в ответе.

        Returns:
            Dict с ключами: content, role, model, finish_reason, usage.
        """
        ...

    @abstractmethod
    async def models(self) -> list[str]:
        """Получить список доступных моделей."""
        ...

    @abstractmethod
    async def health(self) -> bool:
        """Проверить доступность провайдера."""
        ...
