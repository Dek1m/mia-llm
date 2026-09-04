"""OpenAI-compatible LLM Provider — работает с OpenAI API и совместимыми серверами."""
from __future__ import annotations

import time
from typing import Any

from prometheus_client import REGISTRY, Counter, Histogram

from .base import BaseProvider

__all__ = ["OpenAIProvider"]


def _counter(name: str, documentation: str, labelnames: list[str]) -> Counter:
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    return Counter(name, documentation, labelnames)


def _histogram(name: str, documentation: str) -> Histogram:
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    return Histogram(
        name,
        documentation,
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )


llm_chat_total = _counter(
    "llm_chat_total",
    "Total LLM chat completions",
    ["status"],
)
llm_chat_duration_seconds = _histogram(
    "llm_chat_duration_seconds",
    "LLM chat completion duration in seconds",
)


def _tools_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") or {}
        items.append({
            "name": fn.get("name") or raw.get("name") or "tool",
            "args": fn.get("arguments") or "",
            "status": "done",
        })
    return items


def _retry_after(resp: Any, default: float = 2.0) -> float:
    """Секунды из Retry-After; дата-формат и мусор → default, потолок 30с."""
    raw = resp.headers.get("retry-after") if resp is not None else None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return min(max(value, 0.5), 30.0)


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

        on_delta = kwargs.pop("on_delta", None)
        extra = kwargs.pop("extra", None) or {}
        model = model or self._default_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)
        payload.update(kwargs)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"
        started = time.monotonic()
        try:
            if on_delta is not None:
                data = await self._stream_chat(httpx, url, payload, headers, on_delta)
            else:
                data = None
                for attempt in (0, 1):  # один повтор на 429 — лимиты z.ai и подобных
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        resp = await client.post(url, json=payload, headers=headers)
                    if resp.status_code == 429 and attempt == 0:
                        import asyncio

                        await asyncio.sleep(_retry_after(resp))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                if data is None:
                    resp.raise_for_status()
        except Exception as exc:
            duration_s = time.monotonic() - started
            duration_ms = round(duration_s * 1000, 1)
            llm_chat_total.labels(status="error").inc()
            llm_chat_duration_seconds.observe(duration_s)
            if self._log is not None:
                self._log.error(
                    "llm_chat_failed",
                    extra={
                        "llm.model": model,
                        "llm.duration_ms": duration_ms,
                        "error_type": type(exc).__name__,
                    },
                )
            raise

        duration_s = time.monotonic() - started
        duration_ms = round(duration_s * 1000, 1)
        llm_chat_total.labels(status="ok").inc()
        llm_chat_duration_seconds.observe(duration_s)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        log_extra = {
            "llm.model": data.get("model", model),
            "llm.duration_ms": duration_ms,
            "llm.tokens.input": usage.get("prompt_tokens"),
            "llm.tokens.output": usage.get("completion_tokens"),
        }
        if self._log is not None:
            if duration_ms > 500:
                self._log.warning("llm_chat_slow", extra=log_extra)
            else:
                self._log.info("llm_chat_ok", extra=log_extra)

        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or data.get("reasoning")
            or ""
        )
        tools = _tools_from_message(message)
        return {
            "content": message.get("content"),
            "reasoning": reasoning or None,
            "tools": tools,
            "role": message.get("role", "assistant"),
            "model": data.get("model", model),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
        }

    async def _stream_chat(
        self,
        httpx: Any,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        on_delta: Any,
    ) -> dict[str, Any]:
        import json

        from ..loop import stages_from_output

        body = dict(payload)
        body["stream"] = True
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tools: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        model = str(payload.get("model") or "")
        finish = None
        last_emit = 0.0
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    line = raw.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    if chunk.get("model"):
                        model = str(chunk["model"])
                    choice = (chunk.get("choices") or [{}])[0]
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or choice.get("message") or {}
                    piece = delta.get("content")
                    if piece:
                        content_parts.append(str(piece))
                    think = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or ""
                    )
                    if isinstance(delta.get("thinking"), str):
                        think = think or delta.get("thinking")
                    if think:
                        reasoning_parts.append(str(think))
                    for item in delta.get("tool_calls") or []:
                        fn = item.get("function") or {}
                        tools.append({
                            "name": fn.get("name") or item.get("name") or "tool",
                            "args": fn.get("arguments") or "",
                            "status": "running",
                        })
                    now = time.monotonic()
                    if now - last_emit >= 0.12:
                        last_emit = now
                        content = "".join(content_parts)
                        reasoning = "".join(reasoning_parts)
                        trace = {
                            "content": content,
                            "reasoning": reasoning,
                            "stages": stages_from_output(reasoning, content, tools),
                        }
                        emitted = on_delta(trace)
                        if hasattr(emitted, "__await__"):
                            await emitted
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        for item in tools:
            item["status"] = "done"
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": [
                        {"function": {"name": item["name"], "arguments": item["args"]}}
                        for item in tools
                    ],
                },
                "finish_reason": finish,
            }],
            "model": model,
            "usage": usage,
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
                self._log.warning("llm_models_failed", extra={"error_type": type(e).__name__})
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
