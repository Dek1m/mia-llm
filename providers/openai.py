"""OpenAI-compatible LLM Provider — работает с OpenAI API и совместимыми серверами."""
from __future__ import annotations

from typing import Any

from .base import BaseProvider

__all__ = ["OpenAIProvider"]


class OpenAIProvider(BaseProvider):
    """Провайдер для OpenAI-compatible API.

    Поддерживает OpenAI, llama.cpp server, vLLM и другие совместимые бэкенды.
    """

    def __init__(
        self,
        name: str = "openai",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        default_model: str = "gpt-4o-mini",
        timeout: float = 120.0,
        log: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, config=kwargs, log=log)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Выполнить chat completion через OpenAI-compatible API."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed — required for OpenAI provider")

        model = model or self._default_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        return {
            "content": message.get("content"),
            "role": message.get("role", "assistant"),
            "model": data.get("model", model),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
        }

    async def models(self) -> list[str]:
        """Получить список моделей (если API поддерживает)."""
        try:
            import httpx
        except ImportError:
            return []

        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return [m.get("id", "") for m in data.get("data", [])]
        except Exception as e:
            if self._log is not None:
                self._log.warning("Failed to list models", extra={"error": str(e)})
            return []

    async def health(self) -> bool:
        """Проверить доступность API."""
        try:
            import httpx
        except ImportError:
            return False

        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
                return resp.status_code == 200
        except Exception:
            return False
