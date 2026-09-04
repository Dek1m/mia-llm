"""ProviderRegistry — реестр LLM-провайдеров с fallback-логикой."""
from __future__ import annotations

from typing import Any

from .base import BaseProvider

__all__ = ["ProviderRegistry"]


class ProviderRegistry:
    """Реестр LLM-провайдеров.

    Поддерживает:
    - Регистрацию провайдеров по имени
    - Выбор default/fallback
    - Автоматический fallback при ошибках default-провайдера
    """

    def __init__(self, log: Any | None = None) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._default: str | None = None
        self._fallback: str | None = None
        self._log = log

    def register(self, name: str, provider: BaseProvider) -> None:
        """Зарегистрировать провайдер."""
        self._providers[name] = provider
        if self._log is not None:
            self._log.info("provider_registered", name=name)

    def get(self, name: str) -> BaseProvider | None:
        """Получить провайдер по имени."""
        return self._providers.get(name)

    def get_default(self) -> BaseProvider | None:
        """Получить default-провайдер."""
        if self._default and self._default in self._providers:
            return self._providers[self._default]
        # Первый зарегистрированный
        return next(iter(self._providers.values()), None)

    def get_fallback(self) -> BaseProvider | None:
        """Получить fallback-провайдер."""
        if self._fallback and self._fallback in self._providers:
            return self._providers[self._fallback]
        return None

    def set_default(self, name: str) -> None:
        """Установить default-провайдер."""
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")
        self._default = name

    def set_fallback(self, name: str) -> None:
        """Установить fallback-провайдер."""
        if name and name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")
        self._fallback = name

    def list_providers(self) -> list[dict[str, Any]]:
        """Список провайдеров с метаданными."""
        result = []
        for name, provider in self._providers.items():
            result.append({
                "name": name,
                "is_default": name == self._default,
                "is_fallback": name == self._fallback,
            })
        return result

    async def chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Вызвать chat через default провайдер с fallback.

        Логика:
        1. Попытка через default
        2. При ошибке (ConnectionError, TimeoutError, 5xx) → fallback
        3. Если fallback не задан → пробрасываем ошибку
        """
        default = self.get_default()
        if not default:
            raise RuntimeError("No LLM provider registered")

        try:
            return await default.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            # Проверяем тип ошибки — retry только на network/timeout/5xx
            should_retry = _is_retryable_error(e)
            if not should_retry:
                raise

            fallback = self.get_fallback()
            if fallback is None or fallback is default:
                raise

            if self._log is not None:
                self._log.warning(
                    "llm_fallback_triggered",
                    extra={"default": self._default, "fallback": self._fallback, "error": str(e)},
                )
            return await fallback.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

    def has_module(self, name: str) -> bool:
        """Проверить, зарегистрирован ли провайдер."""
        return name in self._providers


def _is_retryable_error(e: Exception) -> bool:
    """Проверить, является ли ошибка ретраибелной."""
    error_type = type(e).__name__
    retryable_types = (
        "ConnectionError",
        "TimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
    )
    if error_type in retryable_types:
        return True

    # HTTPStatusError с кодом 5xx или 429 (лимиты вендора — временно)
    if hasattr(e, "response") and hasattr(e.response, "status_code"):
        status = e.response.status_code
        if 500 <= status < 600 or status == 429:
            return True

    return False
