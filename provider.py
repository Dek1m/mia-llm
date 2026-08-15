"""LLM Provider — абстракция над LLM-бэкендами и agent definitions."""
from __future__ import annotations

import threading
from typing import Any, Iterator
from uuid import uuid4

from argenta_logging import get_logger

try:
    from core.task_decorator import task
except ImportError:
    def task(**kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco

from .config import LLMConfig
from .models import AgentDefinition, ChatMessage, ChatResponse, StreamChunk

log = get_logger(__name__)

__all__ = ["LLMProvider"]


class LLMProvider:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._agents: dict[str, AgentDefinition] = {}
        self._lock = threading.RLock()

    @task(type="io", timeout=5.0)
    def register_agent(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentDefinition:
        agent = AgentDefinition(
            id=agent_id,
            name=name,
            system_prompt=system_prompt,
            model=model or self._config.default_model,
            temperature=temperature if temperature is not None else self._config.default_temperature,
            max_tokens=max_tokens or self._config.default_max_tokens,
            tools=tools or [],
            metadata=metadata or {},
        )
        with self._lock:
            self._agents[agent_id] = agent
        log.info("agent_registered", agent_id=agent_id, name=name)
        return agent

    @task(type="io", timeout=5.0)
    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        with self._lock:
            return self._agents.get(agent_id)

    @task(type="io", timeout=5.0)
    def list_agents(self) -> list[AgentDefinition]:
        with self._lock:
            return list(self._agents.values())

    @task(type="io", timeout=5.0)
    def unregister_agent(self, agent_id: str) -> bool:
        with self._lock:
            return self._agents.pop(agent_id, None) is not None

    @task(type="network", timeout=120.0, retry=1)
    def chat(
        self,
        messages: list[dict[str, Any] | ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        normalized = self._normalize_messages(messages)
        model = model or self._config.default_model
        temperature = temperature if temperature is not None else self._config.default_temperature
        max_tokens = max_tokens or self._config.default_max_tokens

        if self._config.default_provider == "mock":
            return self._mock_chat(normalized, model)

        return self._openai_compatible_chat(
            messages=normalized,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
            **kwargs,
        )

    @task(type="network", timeout=120.0)
    def chat_stream(
        self,
        messages: list[dict[str, Any] | ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        normalized = self._normalize_messages(messages)
        model = model or self._config.default_model
        temperature = temperature if temperature is not None else self._config.default_temperature
        max_tokens = max_tokens or self._config.default_max_tokens

        if self._config.default_provider == "mock":
            yield from self._mock_stream(normalized, model)
            return

        resp = self._openai_compatible_chat(
            messages=normalized,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )
        yield StreamChunk(delta=resp.content or "", finish_reason=resp.finish_reason)

    @task(type="network", timeout=120.0, retry=1)
    def chat_as_agent(
        self,
        agent_id: str,
        messages: list[dict[str, Any] | ChatMessage],
        **overrides: Any,
    ) -> ChatResponse:
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        system = {"role": "system", "content": agent.system_prompt}
        full_messages = [system] + self._normalize_messages(messages)

        return self.chat(
            messages=full_messages,
            model=overrides.get("model") or agent.model,
            temperature=overrides.get("temperature") if "temperature" in overrides else agent.temperature,
            max_tokens=overrides.get("max_tokens") or agent.max_tokens,
            tools=overrides.get("tools"),
        )

    def _normalize_messages(
        self, messages: list[dict[str, Any] | ChatMessage]
    ) -> list[dict[str, Any]]:
        result = []
        for m in messages:
            if isinstance(m, ChatMessage):
                d: dict[str, Any] = {"role": m.role}
                if m.content is not None:
                    d["content"] = m.content
                if m.name:
                    d["name"] = m.name
                if m.tool_calls:
                    d["tool_calls"] = m.tool_calls
                if m.tool_call_id:
                    d["tool_call_id"] = m.tool_call_id
                result.append(d)
            else:
                result.append(dict(m))
        return result

    def _mock_chat(self, messages: list[dict], model: str) -> ChatResponse:
        last = next((m["content"] for m in reversed(messages) if m.get("content")), "")
        return ChatResponse(
            id=f"mock-{uuid4()}",
            content=f"[mock:{model}] Echo: {last[:200]}",
            model=model,
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def _mock_stream(self, messages: list[dict], model: str) -> Iterator[StreamChunk]:
        resp = self._mock_chat(messages, model)
        text = resp.content or ""
        for i in range(0, len(text), 12):
            yield StreamChunk(delta=text[i : i + 12])
        yield StreamChunk(delta="", finish_reason="stop")

    def _openai_compatible_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        try:
            import httpx
        except ImportError:
            log.warning("httpx not installed, falling back to mock")
            return self._mock_chat(messages, model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        url = self._config.base_url.rstrip("/") + "/chat/completions"

        with httpx.Client(timeout=self._config.timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return ChatResponse.from_openai(data)
