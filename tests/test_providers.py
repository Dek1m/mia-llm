"""Tests for LLM Providers — BaseProvider, OpenAIProvider, ProviderRegistry."""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from modules.llm.providers.base import BaseProvider
from modules.llm.providers.openai import OpenAIProvider
from modules.llm.providers.registry import ProviderRegistry


class _Resp:
    def __init__(self, status: int, body: dict[str, Any] | None = None, retry_after: str | None = None) -> None:
        self.status_code = status
        self._body = body or {}
        self.headers = {"retry-after": retry_after} if retry_after else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://vendor.test/v1/chat/completions")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=httpx.Response(self.status_code)
            )

    def json(self) -> dict[str, Any]:
        return self._body


class _RetryClient:
    """Отдаёт очередь ответов: 429 → 200."""

    instances: list[_RetryClient] = []

    def __init__(self, *, responses: list[_Resp], **_kwargs: Any) -> None:
        self.responses = list(responses)
        self.posts: list[dict[str, Any]] = []
        _RetryClient.instances.append(self)

    async def __aenter__(self) -> _RetryClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> _Resp:
        self.posts.append({"url": url, "json": json})
        return self.responses.pop(0)


class FakeLLMProvider:
    """Фейковый LLM-провайдер для тестов."""

    def __init__(self, name: str = "fake", should_fail: bool = False) -> None:
        self._name = name
        self._should_fail = should_fail

    @property
    def name(self) -> str:
        return self._name

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if self._should_fail:
            raise ConnectionError("Fake provider connection error")
        return {
            "content": f"[fake:{self._name}] Echo",
            "role": "assistant",
            "model": "fake-model",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def models(self) -> list[str]:
        return ["fake-model", "fake-model-2"]

    async def health(self) -> bool:
        return not self._should_fail


class TestBaseProvider:
    def test_cannot_instantiate_abstract(self):
        """BaseProvider — ABC, нельзя инстанцировать напрямую."""
        with pytest.raises(TypeError):
            BaseProvider(name="test")


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_health_unreachable(self):
        """health() при недоступном сервере → False."""
        provider = OpenAIProvider(
            name="test",
            base_url="http://localhost:99999/v1",
            api_key="",
            timeout=1.0,
        )
        result = await provider.health()
        assert result is False

    @pytest.mark.asyncio
    async def test_models_unreachable(self):
        """models() при недоступном сервере → пустой список."""
        provider = OpenAIProvider(
            name="test",
            base_url="http://localhost:99999/v1",
            api_key="",
            timeout=1.0,
        )
        result = await provider.models()
        assert result == []

    def test_name_property(self):
        provider = OpenAIProvider(name="my-provider")
        assert provider.name == "my-provider"

    @pytest.mark.asyncio
    async def test_chat_retries_once_on_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ok = _Resp(200, {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], "model": "m"})
        client = _RetryClient(responses=[_Resp(429, retry_after="0"), ok])
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
        provider = OpenAIProvider(name="t", base_url="https://vendor.test/v1", api_key="k")
        result = await provider.chat(messages=[{"role": "user", "content": "1"}])
        assert result["content"] == "hi"
        assert len(client.posts) == 2

    @pytest.mark.asyncio
    async def test_chat_raises_status_error_after_second_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _RetryClient(responses=[_Resp(429, retry_after="0"), _Resp(429, retry_after="0")])
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
        provider = OpenAIProvider(name="t", base_url="https://vendor.test/v1", api_key="k")
        with pytest.raises(httpx.HTTPStatusError):
            await provider.chat(messages=[{"role": "user", "content": "1"}])

    @pytest.mark.asyncio
    async def test_stream_no_double_context_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Регресс: у httpx.Response нет __aenter__ — двойной вход падал TypeError."""

        class _StreamResp:
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self) -> None:
                self.aenter_calls = 0

            def raise_for_status(self) -> None:
                return None

            def __enter__(self):  # намеренно: только sync-CM, как у httpx.Response
                raise AssertionError("sync entry must not be used")

            async def aiter_lines(self):
                import json

                yield 'data: {"choices":[{"delta":{"content":"he"}}]}'
                yield 'data: {"choices":[{"delta":{"content":"y"}}],"usage":{"prompt_tokens":1,"completion_tokens":2}}'
                yield "data: [DONE]"

        class _StreamCtx:
            def __init__(self, resp: _StreamResp) -> None:
                self.resp = resp

            async def __aenter__(self):
                self.resp.aenter_calls += 1
                return self.resp

            async def __aexit__(self, *args: Any):
                return None

        class _StreamClient:
            def __init__(self, **kwargs: Any) -> None:
                self.resp = _StreamResp()
                self.closed = False

            async def aclose(self) -> None:
                self.closed = True

            def stream(self, *args: Any, **kwargs: Any) -> _StreamCtx:
                return _StreamCtx(self.resp)

        client = _StreamClient()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
        provider = OpenAIProvider(name="t", base_url="https://vendor.test/v1", api_key="k")
        seen: list[dict[str, Any]] = []

        async def on_delta(trace: dict[str, Any]) -> None:
            seen.append(trace)

        result = await provider.chat(messages=[{"role": "user", "content": "1"}], on_delta=on_delta)
        assert result["content"] == "hey"
        assert result["usage"]["completion_tokens"] == 2
        assert client.resp.aenter_calls == 1
        assert client.closed is True
        # Эмиты троттлятся (0.12с) — второй чанк может слиться с финалом.
        assert seen and seen[-1]["content"] in ("he", "hey")


class TestProviderRegistry:
    def test_register_and_get(self):
        reg = ProviderRegistry()
        p = FakeLLMProvider(name="test")
        reg.register("test", p)
        assert reg.get("test") is p
        assert reg.get("nonexistent") is None

    def test_default_provider(self):
        reg = ProviderRegistry()
        p1 = FakeLLMProvider(name="p1")
        p2 = FakeLLMProvider(name="p2")
        reg.register("p1", p1)
        reg.register("p2", p2)
        reg.set_default("p1")
        assert reg.get_default() is p1

    def test_fallback_provider(self):
        reg = ProviderRegistry()
        p1 = FakeLLMProvider(name="p1")
        p2 = FakeLLMProvider(name="p2")
        reg.register("p1", p1)
        reg.register("p2", p2)
        reg.set_default("p1")
        reg.set_fallback("p2")
        assert reg.get_fallback() is p2

    def test_set_default_nonexistent_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(ValueError, match="not registered"):
            reg.set_default("nonexistent")

    def test_set_fallback_nonexistent_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(ValueError, match="not registered"):
            reg.set_fallback("nonexistent")

    def test_list_providers(self):
        reg = ProviderRegistry()
        reg.register("p1", FakeLLMProvider(name="p1"))
        reg.register("p2", FakeLLMProvider(name="p2"))
        reg.set_default("p1")
        result = reg.list_providers()
        assert len(result) == 2
        assert any(p["name"] == "p1" and p["is_default"] for p in result)

    @pytest.mark.asyncio
    async def test_chat_with_fallback_success(self):
        reg = ProviderRegistry()
        p = FakeLLMProvider(name="p1")
        reg.register("p1", p)
        reg.set_default("p1")
        result = await reg.chat_with_fallback(messages=[{"role": "user", "content": "hi"}])
        assert result["content"] == "[fake:p1] Echo"

    @pytest.mark.asyncio
    async def test_chat_with_fallback_on_error(self):
        reg = ProviderRegistry()
        p_fail = FakeLLMProvider(name="fail", should_fail=True)
        p_ok = FakeLLMProvider(name="ok")
        reg.register("fail", p_fail)
        reg.register("ok", p_ok)
        reg.set_default("fail")
        reg.set_fallback("ok")
        result = await reg.chat_with_fallback(messages=[{"role": "user", "content": "hi"}])
        assert result["content"] == "[fake:ok] Echo"

    @pytest.mark.asyncio
    async def test_chat_fallback_same_provider_reraises(self):
        """Fallback тот же что default → ошибка не проглатывается."""
        reg = ProviderRegistry()
        p = FakeLLMProvider(name="p1", should_fail=True)
        reg.register("p1", p)
        reg.set_default("p1")
        with pytest.raises(ConnectionError):
            await reg.chat_with_fallback(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_chat_no_providers_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(RuntimeError, match="No LLM provider"):
            await reg.chat_with_fallback(messages=[{"role": "user", "content": "hi"}])

    def test_has_module(self):
        reg = ProviderRegistry()
        reg.register("p1", FakeLLMProvider(name="p1"))
        assert reg.has_module("p1") is True
        assert reg.has_module("nonexistent") is False
